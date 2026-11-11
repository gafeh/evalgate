from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from evalgate.core.grading import evaluate_grading
from evalgate.db.models import Case
from evalgate.db.queue import (
    Claim,
    claim_batch,
    complete_work_item,
    fail_work_item,
    finalize_run_if_complete,
    heartbeat,
    mark_run_running,
    outstanding_count,
    reclaim_stale,
)
from evalgate.db.repositories import record_result, save_judge_assessment
from evalgate.db.session import session_scope
from evalgate.judge.assessor import LlmJudge
from evalgate.runner.hydrate import hydrate_case
from evalgate.runner.protocol import Grader

log = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerStats:
    claimed: int = 0
    succeeded: int = 0
    failed: int = 0
    duplicates: int = 0
    reclaimed: int = 0
    judged: int = 0
    judge_failures: int = 0
    polls: int = 0
    errors: list[str] = field(default_factory=list)


async def _process_claim(
    session: AsyncSession,
    *,
    run_id: UUID,
    claim: Claim,
    grader: Grader,
    judge: LlmJudge | None,
    stats: WorkerStats,
) -> None:
    case_row = await session.get(Case, claim.case_id)
    if case_row is None:
        await fail_work_item(session, claim.work_item_id, error="case row missing")
        stats.failed += 1
        return

    case = hydrate_case(case_row)
    try:
        output = await grader.grade(case)
    except Exception as exc:  # a bad grader must not kill the worker
        status = await fail_work_item(session, claim.work_item_id, error=repr(exc))
        stats.failed += 1
        stats.errors.append(f"{case.id}: {exc!r} -> {status}")
        return

    evaluation = evaluate_grading(case, output.grading) if output.grading is not None else None
    duplicate = False
    result_id: UUID | None = None
    try:
        result = await record_result(
            session,
            run_id=run_id,
            case_id=claim.case_id,
            raw_output=output.raw_output,
            evaluation=evaluation,
            grading=output.grading.model_dump(mode="json") if output.grading else None,
            parse_error=output.parse_error,
            latency_ms=output.latency_ms,
        )
        result_id = result.id
    except IntegrityError:
        # A previous attempt already wrote this result. The unique constraint
        # on (run_id, case_id) is what makes at-least-once delivery safe.
        await session.rollback()
        duplicate = True
        stats.duplicates += 1

    if judge is not None and output.grading is not None and not duplicate and result_id:
        try:
            assessed = await judge.assess(case, output.grading)
        except Exception as exc:  # the judge is instrumentation, not the job
            assessed = None
            stats.errors.append(f"judge {case.id}: {exc!r}")
        if assessed is None:
            stats.judge_failures += 1
        else:
            await save_judge_assessment(
                session,
                result_id=result_id,
                model=assessed.model,
                assessment=assessed.assessment,
                cache_key=assessed.cache_key,
            )
            stats.judged += 1

    await complete_work_item(session, claim.work_item_id)
    stats.succeeded += 1


async def process_batch(
    session: AsyncSession,
    *,
    run_id: UUID,
    grader: Grader,
    worker_id: str,
    batch_size: int,
    stats: WorkerStats,
    judge: LlmJudge | None = None,
) -> int:
    claims = await claim_batch(
        session, run_id=run_id, worker_id=worker_id, batch_size=batch_size
    )
    if not claims:
        return 0

    stats.claimed += len(claims)
    remaining = [claim.work_item_id for claim in claims]
    for claim in claims:
        await _process_claim(
            session,
            run_id=run_id,
            claim=claim,
            grader=grader,
            judge=judge,
            stats=stats,
        )
        remaining.remove(claim.work_item_id)
        await heartbeat(session, work_item_ids=remaining, worker_id=worker_id)
    return len(claims)


async def worker_loop(
    engine: AsyncEngine,
    *,
    run_id: UUID,
    grader: Grader,
    worker_id: str,
    batch_size: int = 8,
    poll_interval: float = 0.5,
    idle_limit: int = 3,
    stale_after_seconds: float = 30.0,
    judge: LlmJudge | None = None,
) -> WorkerStats:
    """Claim, grade, record, repeat. Exits when the run has nothing outstanding."""
    stats = WorkerStats()
    idle_rounds = 0

    async with session_scope(engine) as session:
        await mark_run_running(session, run_id)

        while True:
            stats.polls += 1
            stats.reclaimed += await reclaim_stale(
                session, run_id=run_id, stale_after_seconds=stale_after_seconds
            )

            processed = await process_batch(
                session,
                run_id=run_id,
                grader=grader,
                worker_id=worker_id,
                batch_size=batch_size,
                stats=stats,
                judge=judge,
            )
            if processed:
                idle_rounds = 0
                continue

            if await outstanding_count(session, run_id) == 0:
                await finalize_run_if_complete(session, run_id)
                return stats

            idle_rounds += 1
            if idle_rounds >= idle_limit:
                log.info("worker %s idle with work outstanding, exiting", worker_id)
                return stats
            await asyncio.sleep(poll_interval)