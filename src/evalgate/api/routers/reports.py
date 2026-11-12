from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from evalgate.analysis.load import load_label_pairs
from evalgate.api.deps import SessionDep
from evalgate.api.schemas import BinOut, CalibrationOut
from evalgate.core.calibration import compute_calibration

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/calibration", response_model=CalibrationOut)
async def calibration(
    session: SessionDep,
    run_id: UUID | None = None,
    bins: int = 5,
    target: float = 0.90,
    include_triage: bool = False,
) -> CalibrationOut:
    """Judge calibration against human labels.

    Triaged items are excluded by default: they were queued because something
    looked wrong, so they are not a random sample of the judge's work.
    """
    pairs = await load_label_pairs(
        session, run_id=run_id, calibration_only=not include_triage
    )
    report = compute_calibration(pairs, bins=bins, target_accuracy=target)
    return CalibrationOut(
        pairs=report.pairs,
        distinct_cases=report.distinct_cases,
        agreement=report.agreement,
        kappa=report.kappa,
        brier=report.brier,
        expected_calibration_error=report.expected_calibration_error,
        mean_confidence=report.mean_confidence,
        overconfident=report.overconfident,
        trustworthy=report.trustworthy,
        warnings=list(report.warnings),
        bins=[
            BinOut(
                low=item.low,
                high=item.high,
                count=item.count,
                mean_confidence=item.mean_confidence,
                accuracy=item.accuracy,
                gap=item.gap,
            )
            for item in report.bins
        ],
        agreement_by_judge_verdict=dict(report.agreement_by_judge_verdict),
        recommended_threshold=report.recommended_threshold,
        coverage_at_threshold=report.coverage_at_threshold,
        accuracy_at_threshold=report.accuracy_at_threshold,
    )