from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, exists, func, select, update
from sqlalchemy import case as sql_case
from sqlalchemy.ext.asyncio import AsyncSession

from evalgate.db.models import Run, RunStatus, WorkItem, WorkItemStatus

ACTIVE_STATUSES = (WorkItemStatus.QUEUED, WorkItemStatus.CLAIMED)


@dataclass(frozen=True, slots=True)
class Claim:
    work_item_id: UUID
    case_id: UUID
    attempts: int


def _seconds(value: float) -> object:
    """Postgres interval from a float, parameterised rather than interpolated."""
    return func.make_interval(0, 0, 0, 0, 0, 0, value)


async def claim_batch(
    session: AsyncSession,
    *,
    run_id: UUID,
    worker_id: str,
    batch_size: int,
) -> list[Claim]:
    """Atomically take up to batch_size claimable items for this worker.

    The inner SELECT takes row locks and skips anything another worker is
    already looking at, so two workers polling at the same moment get
    disjoint batches instead of one of them blocking.
    """
    claimable = (
        select(WorkItem.id)
        .where(
            WorkItem.run_id == run_id,
            WorkItem.status == WorkItemStatus.QUEUED,
            WorkItem.available_at <= func.now(),
        )
        .order_by(WorkItem.available_at, WorkItem.created_at)
        .limit(batch_size)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )

    stmt = (
        update(WorkItem)
        .where(WorkItem.id.in_(claimable))
        .values(
            status=WorkItemStatus.CLAIMED.value,
            claimed_by=worker_id,
            claimed_at=func.now(),
            heartbeat_at=func.now(),
            attempts=WorkItem.attempts + 1,
        )
        .returning(WorkItem.id, WorkItem.case_id, WorkItem.attempts)
        .execution_options(synchronize_session=False)
    )
    rows = (await session.execute(stmt)).all()
    # A claim that is not committed is not a claim: the locks would drop
    # and another worker could take the same rows.
    await session.commit()
    return [Claim(work_item_id=row[0], case_id=row[1], attempts=row[2]) for row in rows]


async def heartbeat(
    session: AsyncSession, *, work_item_ids: list[UUID], worker_id: str
) -> None:
    if not work_item_ids:
        return
    stmt = (
        update(WorkItem)
        .where(
            WorkItem.id.in_(work_item_ids),
            WorkItem.claimed_by == worker_id,
            WorkItem.status == WorkItemStatus.CLAIMED,
        )
        .values(heartbeat_at=func.now())
        .execution_options(synchronize_session=False)
    )
    await session.execute(stmt)
    await session.commit()


async def complete_work_item(session: AsyncSession, work_item_id: UUID) -> None:
    stmt = (
        update(WorkItem)
        .where(WorkItem.id == work_item_id)
        .values(
            status=WorkItemStatus.DONE.value,
            claimed_by=None,
            claimed_at=None,
            heartbeat_at=None,
            last_error=None,
        )
        .execution_options(synchronize_session=False)
    )
    await session.execute(stmt)
    await session.commit()


async def fail_work_item(
    session: AsyncSession,
    work_item_id: UUID,
    *,
    error: str,
    backoff_seconds: float = 5.0,
) -> WorkItemStatus:
    """Record a failure. Requeue with backoff, or bury it once attempts run out."""
    stmt = (
        update(WorkItem)
        .where(WorkItem.id == work_item_id)
        .values(
            status=sql_case(
                (
                    WorkItem.attempts >= WorkItem.max_attempts,
                    WorkItemStatus.DEAD.value,
                ),
                else_=WorkItemStatus.QUEUED.value,
            ),
            claimed_by=None,
            claimed_at=None,
            heartbeat_at=None,
            available_at=func.now() + _seconds(backoff_seconds),
            last_error=error[:2000],
        )
        .returning(WorkItem.status)
        .execution_options(synchronize_session=False)
    )
    status = (await session.execute(stmt)).scalar_one()
    await session.commit()
    return WorkItemStatus(status)


async def reclaim_stale(
    session: AsyncSession, *, run_id: UUID, stale_after_seconds: float
) -> int:
    """Return items whose worker stopped reporting. Bury the ones out of attempts.

    Staleness is measured against the database clock, not the worker's,
    because the whole point is that the worker may no longer exist.
    """
    cutoff = WorkItem.heartbeat_at < func.now() - _seconds(stale_after_seconds)

    bury = (
        update(WorkItem)
        .where(
            WorkItem.run_id == run_id,
            WorkItem.status == WorkItemStatus.CLAIMED,
            cutoff,
            WorkItem.attempts >= WorkItem.max_attempts,
        )
        .values(
            status=WorkItemStatus.DEAD.value,
            last_error="worker stopped sending heartbeats",
            claimed_by=None,
            claimed_at=None,
            heartbeat_at=None,
        )
        .execution_options(synchronize_session=False)
    )
    await session.execute(bury)

    requeue = (
        update(WorkItem)
        .where(
            WorkItem.run_id == run_id,
            WorkItem.status == WorkItemStatus.CLAIMED,
            cutoff,
            WorkItem.attempts < WorkItem.max_attempts,
        )
        .values(
            status=WorkItemStatus.QUEUED.value,
            claimed_by=None,
            claimed_at=None,
            heartbeat_at=None,
            available_at=func.now(),
            last_error="reclaimed after worker went silent",
        )
        .returning(WorkItem.id)
        .execution_options(synchronize_session=False)
    )
    reclaimed = (await session.execute(requeue)).scalars().all()
    await session.commit()
    return len(reclaimed)


async def outstanding_count(session: AsyncSession, run_id: UUID) -> int:
    stmt = select(func.count()).where(
        WorkItem.run_id == run_id, WorkItem.status.in_(ACTIVE_STATUSES)
    )
    return int((await session.execute(stmt)).scalar_one())


async def mark_run_running(session: AsyncSession, run_id: UUID) -> None:
    stmt = (
        update(Run)
        .where(Run.id == run_id, Run.status == RunStatus.PENDING)
        .values(status=RunStatus.RUNNING.value, started_at=func.now())
        .execution_options(synchronize_session=False)
    )
    await session.execute(stmt)
    await session.commit()


async def finalize_run_if_complete(session: AsyncSession, run_id: UUID) -> bool:
    """Flip the run to completed, in one statement, only if nothing is outstanding.

    Doing the check and the write as separate steps would let two workers
    both see an empty queue and both finish the run.
    """
    still_working = exists(
        select(WorkItem.id).where(
            and_(WorkItem.run_id == run_id, WorkItem.status.in_(ACTIVE_STATUSES))
        )
    )
    stmt = (
        update(Run)
        .where(
            Run.id == run_id,
            Run.status.in_((RunStatus.PENDING, RunStatus.RUNNING)),
            ~still_working,
        )
        .values(status=RunStatus.COMPLETED.value, finished_at=func.now())
        .returning(Run.id)
        .execution_options(synchronize_session=False)
    )
    changed = (await session.execute(stmt)).scalar_one_or_none()
    await session.commit()
    return changed is not None