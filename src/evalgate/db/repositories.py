from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalgate.core.dataset import Dataset as CoreDataset
from evalgate.core.grading import GradingEvaluation
from evalgate.core.hashing import content_hash
from evalgate.core.judgment import JudgeAssessment
from evalgate.db.models import (
    Case,
    Dataset,
    JudgeAssessmentRow,
    Result,
    Run,
    RunStatus,
    WorkItem,
    WorkItemStatus,
)


class DatasetConflictError(RuntimeError):
    """A dataset with this name and version exists with different content."""


async def get_dataset(session: AsyncSession, name: str, version: str) -> Dataset | None:
    stmt = select(Dataset).where(Dataset.name == name, Dataset.version == version)
    return (await session.execute(stmt)).scalar_one_or_none()


async def save_dataset(session: AsyncSession, dataset: CoreDataset) -> Dataset:
    """Store a dataset and its cases. Idempotent for identical content.

    Re-saving the same name and version with changed content is an error
    rather than an update, because silently mutating a dataset invalidates
    every run that already pointed at it.
    """
    digest = dataset.content_hash()
    existing = await get_dataset(session, dataset.name, dataset.version)
    if existing is not None:
        if existing.content_hash != digest:
            raise DatasetConflictError(
                f"{dataset.name}@{dataset.version} is already stored with different "
                f"content ({existing.content_hash[:12]} vs {digest[:12]}). Bump the version."
            )
        return existing

    row = Dataset(
        name=dataset.name,
        version=dataset.version,
        domain=dataset.domain,
        content_hash=digest,
        case_count=len(dataset.cases),
    )
    session.add(row)
    await session.flush()

    session.add_all(
        [
            Case(
                dataset_id=row.id,
                external_id=case.id,
                question_id=case.question_id,
                question_prompt=case.question_prompt,
                student_response=case.student_response,
                rubric=case.rubric.model_dump(mode="json"),
                expected=case.expected.model_dump(mode="json"),
                tags=dict(case.tags),
            )
            for case in dataset.cases
        ]
    )
    await session.commit()
    return row


async def list_cases(session: AsyncSession, dataset_id: UUID) -> Sequence[Case]:
    stmt = select(Case).where(Case.dataset_id == dataset_id).order_by(Case.external_id)
    return (await session.execute(stmt)).scalars().all()


async def create_run(
    session: AsyncSession,
    *,
    dataset: Dataset,
    model: str,
    params: dict[str, Any] | None = None,
    label: str | None = None,
) -> Run:
    """Create a run and enqueue one work item per case, in one transaction."""
    run = Run(
        dataset_id=dataset.id,
        dataset_content_hash=dataset.content_hash,
        model=model,
        params=params or {},
        label=label,
        status=RunStatus.PENDING,
    )
    session.add(run)
    await session.flush()

    cases = await list_cases(session, dataset.id)
    session.add_all([WorkItem(run_id=run.id, case_id=case.id) for case in cases])
    await session.commit()
    return run


async def get_run(session: AsyncSession, run_id: UUID) -> Run | None:
    return await session.get(Run, run_id)


async def work_item_counts(session: AsyncSession, run_id: UUID) -> dict[str, int]:
    stmt = (
        select(WorkItem.status, func.count())
        .where(WorkItem.run_id == run_id)
        .group_by(WorkItem.status)
    )
    rows = (await session.execute(stmt)).all()
    return {str(status): count for status, count in rows}


async def record_result(
    session: AsyncSession,
    *,
    run_id: UUID,
    case_id: UUID,
    raw_output: str,
    evaluation: GradingEvaluation | None = None,
    grading: dict[str, Any] | None = None,
    parse_error: str | None = None,
    latency_ms: int | None = None,
) -> Result:
    """Persist one scored case. evaluation is None when the output never parsed."""
    scores = (
        {score.scorer: {"value": score.value, "passed": score.passed, "detail": score.detail}
         for score in evaluation.scores}
        if evaluation
        else {}
    )
    payload: dict[str, Any] = {
        "run_id": str(run_id),
        "case_id": str(case_id),
        "raw_output": raw_output,
        "grading": grading,
        "scores": scores,
        "overall": evaluation.overall if evaluation else None,
    }

    result = Result(
        run_id=run_id,
        case_id=case_id,
        raw_output=raw_output,
        parsed_ok=parse_error is None and grading is not None,
        parse_error=parse_error,
        grading=grading,
        scores=scores,
        overall=evaluation.overall if evaluation else None,
        over_credit_points=evaluation.over_credit_points if evaluation else None,
        under_credit_points=evaluation.under_credit_points if evaluation else None,
        hallucinated_item_ids={
            "ids": list(evaluation.hallucinated_item_ids) if evaluation else []
        },
        latency_ms=latency_ms,
        content_hash=content_hash(payload),
    )
    session.add(result)
    await session.commit()
    return result


async def list_results(session: AsyncSession, run_id: UUID) -> Sequence[Result]:
    stmt = select(Result).where(Result.run_id == run_id).order_by(Result.created_at)
    return (await session.execute(stmt)).scalars().all()


async def mark_work_item_done(session: AsyncSession, work_item_id: UUID) -> None:
    item = await session.get(WorkItem, work_item_id)
    if item is None:
        raise LookupError(f"no work item {work_item_id}")
    item.status = WorkItemStatus.DONE
    await session.commit()
async def save_judge_assessment(
    session: AsyncSession,
    *,
    result_id: UUID,
    model: str,
    assessment: JudgeAssessment,
    cache_key: str,
) -> JudgeAssessmentRow:
    row = JudgeAssessmentRow(
        result_id=result_id,
        model=model,
        verdict=assessment.verdict,
        confidence=assessment.confidence,
        reasoning=assessment.reasoning,
        criteria=dict(assessment.criteria),
        cache_key=cache_key,
    )
    session.add(row)
    await session.commit()
    return row


async def get_judge_assessment(
    session: AsyncSession, result_id: UUID
) -> JudgeAssessmentRow | None:
    stmt = select(JudgeAssessmentRow).where(JudgeAssessmentRow.result_id == result_id)
    return (await session.execute(stmt)).scalar_one_or_none()
