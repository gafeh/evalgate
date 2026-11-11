from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalgate.core.calibration import LabelPair
from evalgate.core.judgment import JudgeVerdict
from evalgate.core.outcome import CaseOutcome
from evalgate.db.models import (
    Case,
    HumanLabel,
    JudgeAssessmentRow,
    Result,
    ReviewItem,
    ReviewReason,
)


async def load_outcomes(session: AsyncSession, run_id: UUID) -> list[CaseOutcome]:
    """Flatten a run's results into pure analysis records.

    Cases are keyed by their dataset-level external id rather than row id, so
    two runs over the same dataset version pair up correctly.
    """
    stmt = (
        select(Result, Case.external_id, Case.tags)
        .join(Case, Case.id == Result.case_id)
        .where(Result.run_id == run_id)
        .order_by(Case.external_id)
    )
    rows = (await session.execute(stmt)).all()

    outcomes: list[CaseOutcome] = []
    for result, external_id, tags in rows:
        over = result.over_credit_points or 0.0
        under = result.under_credit_points or 0.0
        outcomes.append(
            CaseOutcome(
                case_id=external_id,
                parsed_ok=result.parsed_ok,
                overall=result.overall,
                over_credit_points=over,
                under_credit_points=under,
                exact_match=result.parsed_ok and over == 0.0 and under == 0.0,
                tags={str(k): str(v) for k, v in (tags or {}).items()},
            )
        )
    return outcomes
async def load_label_pairs(
    session: AsyncSession,
    *,
    run_id: UUID | None = None,
    calibration_only: bool = True,
) -> list[LabelPair]:
    """Judge verdict beside human verdict, for every labelled case.

    calibration_only defaults to True on purpose. Triaged items were queued
    because something looked wrong, so measuring judge accuracy on them
    would answer a question nobody asked.
    """
    stmt = (
        select(
            Case.external_id,
            JudgeAssessmentRow.verdict,
            JudgeAssessmentRow.confidence,
            HumanLabel.verdict,
        )
        .join(ReviewItem, ReviewItem.id == HumanLabel.review_item_id)
        .join(Result, Result.id == ReviewItem.result_id)
        .join(Case, Case.id == Result.case_id)
        .join(JudgeAssessmentRow, JudgeAssessmentRow.result_id == Result.id)
        .select_from(HumanLabel)
        .order_by(Case.external_id)
    )
    if calibration_only:
        stmt = stmt.where(ReviewItem.reason == ReviewReason.CALIBRATION)
    if run_id is not None:
        stmt = stmt.where(Result.run_id == run_id)

    rows = (await session.execute(stmt)).all()
    return [
        LabelPair(
            case_id=external_id,
            judge_verdict=JudgeVerdict(judge_verdict),
            judge_confidence=float(confidence),
            human_verdict=JudgeVerdict(human_verdict),
        )
        for external_id, judge_verdict, confidence, human_verdict in rows
    ]