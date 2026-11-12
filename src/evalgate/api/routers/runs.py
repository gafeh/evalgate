from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from evalgate.analysis.load import load_outcomes
from evalgate.api.deps import SessionDep
from evalgate.api.schemas import (
    CompareOut,
    CreateRunIn,
    IntervalOut,
    ResultOut,
    RunDetailOut,
    RunOut,
    SliceOut,
    SummaryOut,
    UsageOut,
)
from evalgate.core.aggregate import summarize, summarize_by_slice
from evalgate.core.regression import compare_runs
from evalgate.core.stats import ConfidenceInterval
from evalgate.db.models import Case, JudgeAssessmentRow, Result
from evalgate.db.repositories import (
    create_run,
    get_dataset,
    get_run,
    list_runs,
    work_item_counts,
)
from evalgate.judge.cache import run_usage

router = APIRouter(prefix="/runs", tags=["runs"])


def _interval(value: ConfidenceInterval | None) -> IntervalOut | None:
    if value is None:
        return None
    return IntervalOut(
        point=value.point, low=value.low, high=value.high, confidence=value.confidence
    )


@router.get("", response_model=list[RunOut])
async def index(session: SessionDep, limit: int = 100) -> list[RunOut]:
    rows = await list_runs(session, limit=limit)
    return [RunOut.model_validate(row, from_attributes=True) for row in rows]


@router.post("", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
async def create(payload: CreateRunIn, session: SessionDep) -> RunOut:
    """Create a run and enqueue its work. Grading happens in the workers."""
    dataset = await get_dataset(session, payload.dataset_name, payload.dataset_version)
    if dataset is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no dataset {payload.dataset_name}@{payload.dataset_version}",
        )
    run = await create_run(
        session,
        dataset=dataset,
        model=payload.model,
        params=payload.params,
        label=payload.label,
    )
    return RunOut.model_validate(run, from_attributes=True)


@router.get("/{run_id}", response_model=RunDetailOut)
async def show(run_id: UUID, session: SessionDep) -> RunDetailOut:
    run = await get_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no run {run_id}")

    outcomes = await load_outcomes(session, run_id)
    summary = summarize(outcomes)
    usage = await run_usage(session, run_id)

    slices = [
        SliceOut(
            tag_value=key,
            cases=value.cases,
            mean_overall=value.mean_overall,
            over_credit_points=value.over_credit_points,
        )
        for key, value in summarize_by_slice(outcomes, "error_mode").items()
    ]

    return RunDetailOut(
        run=RunOut.model_validate(run, from_attributes=True),
        work_items=await work_item_counts(session, run_id),
        summary=SummaryOut(
            cases=summary.cases,
            unparseable=summary.unparseable,
            mean_overall=summary.mean_overall,
            exact_match_rate=summary.exact_match_rate,
            over_credit_points=summary.over_credit_points,
            under_credit_points=summary.under_credit_points,
        ),
        usage=UsageOut(
            calls=usage.calls,
            cache_hits=usage.cache_hits,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
        ),
        slices=slices,
    )


@router.get("/{run_id}/results", response_model=list[ResultOut])
async def results(run_id: UUID, session: SessionDep) -> list[ResultOut]:
    stmt = (
        select(Result, Case.external_id, JudgeAssessmentRow)
        .join(Case, Case.id == Result.case_id)
        .outerjoin(JudgeAssessmentRow, JudgeAssessmentRow.result_id == Result.id)
        .where(Result.run_id == run_id)
        .order_by(Case.external_id)
    )
    rows = (await session.execute(stmt)).all()
    return [
        ResultOut(
            case_id=external_id,
            parsed_ok=result.parsed_ok,
            overall=result.overall,
            over_credit_points=result.over_credit_points,
            under_credit_points=result.under_credit_points,
            judge_verdict=str(assessment.verdict) if assessment else None,
            judge_confidence=assessment.confidence if assessment else None,
            feedback=str((result.grading or {}).get("feedback", "")),
        )
        for result, external_id, assessment in rows
    ]


@router.get("/{run_id}/compare/{other_id}", response_model=CompareOut)
async def compare(run_id: UUID, other_id: UUID, session: SessionDep) -> CompareOut:
    """Gate run_id (candidate) against other_id (baseline)."""
    for identifier in (run_id, other_id):
        if await get_run(session, identifier) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no run {identifier}")

    baseline = await load_outcomes(session, other_id)
    candidate = await load_outcomes(session, run_id)
    result = compare_runs(baseline, candidate)

    def summary(value: object) -> SummaryOut:
        return SummaryOut.model_validate(value, from_attributes=True)

    return CompareOut(
        verdict=str(result.verdict),
        reason=result.reason,
        passed=result.passed,
        paired_cases=result.paired_cases,
        baseline=summary(result.baseline),
        candidate=summary(result.candidate),
        overall_delta=_interval(result.overall_delta),
        over_credit_delta=_interval(result.over_credit_delta),
        cases_needed=result.cases_needed,
    )