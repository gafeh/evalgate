from __future__ import annotations

import random
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalgate.core.dataset import ExpectedGrading, Rubric
from evalgate.core.judgment import CRITERIA, JudgeVerdict
from evalgate.db.models import (
    Case,
    HumanLabel,
    JudgeAssessmentRow,
    Result,
    ReviewItem,
    ReviewReason,
    ReviewStatus,
)

UNPARSEABLE_PRIORITY = 3.0
SAFETY_PRIORITY = 2.0
FAIL_PRIORITY = 1.5

TRIAGE_REASONS = (
    ReviewReason.UNPARSEABLE,
    ReviewReason.SAFETY,
    ReviewReason.LOW_CONFIDENCE,
)


@dataclass(frozen=True, slots=True)
class ReviewTask:
    """Everything a human needs to label one result, and nothing more.

    There is deliberately no judge verdict or confidence on this type. The
    labels have to be independent of the judge's opinion or the agreement
    number computed from them is circular.
    """

    review_item_id: UUID
    reason: ReviewReason
    case_id: str
    question_prompt: str
    rubric_lines: tuple[str, ...]
    student_response: str
    expected_awards: tuple[str, ...]
    expected_points: float
    grader_awards: tuple[str, ...]
    grader_points: float
    grader_feedback: str
    must_mention: tuple[str, ...]
    must_not_mention: tuple[str, ...]
    parsed_ok: bool
    raw_output: str


async def _unqueued_results(session: AsyncSession, run_id: UUID) -> list[UUID]:
    stmt = (
        select(Result.id)
        .join(JudgeAssessmentRow, JudgeAssessmentRow.result_id == Result.id)
        .where(
            Result.run_id == run_id,
            Result.id.not_in(select(ReviewItem.result_id)),
        )
        .order_by(Result.created_at)
    )
    return list((await session.execute(stmt)).scalars().all())


async def count_items(
    session: AsyncSession,
    *,
    run_id: UUID | None = None,
    reasons: tuple[ReviewReason, ...] | None = None,
    status: ReviewStatus | None = None,
) -> int:
    stmt = select(func.count()).select_from(ReviewItem)
    if run_id is not None:
        stmt = stmt.join(Result, Result.id == ReviewItem.result_id).where(
            Result.run_id == run_id
        )
    if reasons is not None:
        stmt = stmt.where(ReviewItem.reason.in_(reasons))
    if status is not None:
        stmt = stmt.where(ReviewItem.status == status)
    return int((await session.execute(stmt)).scalar_one())


async def enqueue_calibration_sample(
    session: AsyncSession, *, run_id: UUID, size: int, seed: int = 0
) -> int:
    """Top the calibration sample up to size. Returns how many were added.

    size is a target, not an increment, so running this twice does not grow
    the sample. Call it before enqueue_triage: drawn second, the sample would
    only contain what triage did not want, which is not a random sample.
    """
    existing = await count_items(
        session, run_id=run_id, reasons=(ReviewReason.CALIBRATION,)
    )
    wanted = size - existing
    if wanted <= 0:
        return 0

    candidates = await _unqueued_results(session, run_id)
    if not candidates:
        return 0

    rng = random.Random(seed)
    chosen = rng.sample(candidates, k=min(wanted, len(candidates)))
    session.add_all(
        [
            ReviewItem(
                result_id=result_id,
                reason=ReviewReason.CALIBRATION,
                priority=0.0,
            )
            for result_id in chosen
        ]
    )
    await session.commit()
    return len(chosen)


async def enqueue_triage(
    session: AsyncSession, *, run_id: UUID, confidence_threshold: float = 0.80
) -> int:
    """Queue results that look wrong, most alarming first."""
    stmt = (
        select(
            Result.id,
            Result.parsed_ok,
            Result.over_credit_points,
            JudgeAssessmentRow.verdict,
            JudgeAssessmentRow.confidence,
        )
        .outerjoin(JudgeAssessmentRow, JudgeAssessmentRow.result_id == Result.id)
        .where(
            Result.run_id == run_id,
            Result.id.not_in(select(ReviewItem.result_id)),
        )
        .order_by(Result.created_at)
    )
    rows = (await session.execute(stmt)).all()

    items: list[ReviewItem] = []
    for result_id, parsed_ok, over_credit, verdict, confidence in rows:
        conf = float(confidence) if confidence is not None else 0.0
        if not parsed_ok:
            reason, priority = ReviewReason.UNPARSEABLE, UNPARSEABLE_PRIORITY
        elif (over_credit or 0.0) > 0.0:
            reason, priority = ReviewReason.SAFETY, SAFETY_PRIORITY + (1.0 - conf)
        elif verdict is not None and verdict != JudgeVerdict.PASS:
            reason, priority = ReviewReason.LOW_CONFIDENCE, FAIL_PRIORITY + (1.0 - conf)
        elif confidence is not None and conf < confidence_threshold:
            reason, priority = ReviewReason.LOW_CONFIDENCE, 1.0 - conf
        else:
            continue
        items.append(ReviewItem(result_id=result_id, reason=reason, priority=priority))

    if items:
        session.add_all(items)
        await session.commit()
    return len(items)


async def next_review_task(
    session: AsyncSession,
    *,
    run_id: UUID | None = None,
    reasons: tuple[ReviewReason, ...] | None = None,
) -> ReviewTask | None:
    """Highest-priority pending item, hydrated for display.

    reasons narrows which population to serve. A labelling sitting aimed at
    calibration should pass the calibration reason only, since labelling a
    triaged item contributes nothing to the calibration estimate.

    The id is the final tiebreak because calibration items share a priority
    and are inserted in one transaction, so they share a created_at too.
    Without it the queue order shifts between queries.
    """
    stmt = (
        select(ReviewItem, Result, Case)
        .join(Result, Result.id == ReviewItem.result_id)
        .join(Case, Case.id == Result.case_id)
        .where(ReviewItem.status == ReviewStatus.PENDING)
        .order_by(ReviewItem.priority.desc(), ReviewItem.created_at, ReviewItem.id)
        .limit(1)
    )
    if run_id is not None:
        stmt = stmt.where(Result.run_id == run_id)
    if reasons is not None:
        stmt = stmt.where(ReviewItem.reason.in_(reasons))

    row = (await session.execute(stmt)).first()
    if row is None:
        return None

    item, result, case = row
    rubric = Rubric.model_validate(case.rubric)
    expected = ExpectedGrading.model_validate(case.expected)
    grading = result.grading or {}

    return ReviewTask(
        review_item_id=item.id,
        reason=item.reason,
        case_id=case.external_id,
        question_prompt=case.question_prompt,
        rubric_lines=tuple(
            f"{entry.id} ({entry.points:g} pt): {entry.description}"
            for entry in rubric.items
        ),
        student_response=case.student_response,
        expected_awards=tuple(sorted(expected.awarded_item_ids)),
        expected_points=rubric.points_for(expected.awarded_item_ids),
        grader_awards=tuple(sorted(grading.get("awarded_item_ids", []))),
        grader_points=float(grading.get("total_points", 0.0)),
        grader_feedback=str(grading.get("feedback", "")),
        must_mention=expected.must_mention,
        must_not_mention=expected.must_not_mention,
        parsed_ok=result.parsed_ok,
        raw_output=result.raw_output,
    )


async def save_label(
    session: AsyncSession,
    *,
    review_item_id: UUID,
    labeller: str,
    verdict: JudgeVerdict,
    criteria: dict[str, bool] | None = None,
    notes: str = "",
    seconds_spent: int = 0,
) -> HumanLabel:
    item = await session.get(ReviewItem, review_item_id)
    if item is None:
        raise LookupError(f"no review item {review_item_id}")

    label = HumanLabel(
        review_item_id=review_item_id,
        labeller=labeller,
        verdict=verdict,
        criteria={key: bool((criteria or {}).get(key, False)) for key in CRITERIA},
        notes=notes,
        seconds_spent=seconds_spent,
    )
    session.add(label)
    item.status = ReviewStatus.LABELLED
    await session.commit()
    return label


async def skip_review_item(session: AsyncSession, review_item_id: UUID) -> None:
    item = await session.get(ReviewItem, review_item_id)
    if item is None:
        raise LookupError(f"no review item {review_item_id}")
    item.status = ReviewStatus.SKIPPED
    await session.commit()


async def review_counts(
    session: AsyncSession, *, run_id: UUID | None = None
) -> dict[str, int]:
    stmt = select(ReviewItem.status, ReviewItem.reason, func.count()).group_by(
        ReviewItem.status, ReviewItem.reason
    )
    if run_id is not None:
        stmt = stmt.join(Result, Result.id == ReviewItem.result_id).where(
            Result.run_id == run_id
        )
    rows = (await session.execute(stmt)).all()
    return {f"{status}/{reason}": count for status, reason, count in rows}