from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from evalgate.api.deps import SessionDep
from evalgate.api.schemas import EnqueueIn, EnqueueOut, LabelIn, ReviewTaskOut
from evalgate.core.judgment import JudgeVerdict
from evalgate.db.models import ReviewReason
from evalgate.db.review import (
    TRIAGE_REASONS,
    count_items,
    enqueue_calibration_sample,
    enqueue_triage,
    next_review_task,
    review_counts,
    save_label,
    skip_review_item,
)

router = APIRouter(tags=["review"])

REASON_SETS: dict[str, tuple[ReviewReason, ...] | None] = {
    "calibration": (ReviewReason.CALIBRATION,),
    "triage": TRIAGE_REASONS,
    "all": None,
}


@router.post("/runs/{run_id}/review", response_model=EnqueueOut)
async def enqueue(run_id: UUID, payload: EnqueueIn, session: SessionDep) -> EnqueueOut:
    added = await enqueue_calibration_sample(
        session, run_id=run_id, size=payload.sample, seed=payload.seed
    )
    triaged = await enqueue_triage(session, run_id=run_id) if payload.triage else 0
    return EnqueueOut(
        added_to_sample=added,
        triaged=triaged,
        queue=await review_counts(session, run_id=run_id),
    )


@router.get("/review/next", response_model=ReviewTaskOut)
async def next_task(
    session: SessionDep, run_id: UUID | None = None, reason: str = "calibration"
) -> ReviewTaskOut:
    """The next item to label. Carries no judge verdict or confidence."""
    if reason not in REASON_SETS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"unknown reason '{reason}'"
        )
    task = await next_review_task(session, run_id=run_id, reasons=REASON_SETS[reason])
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "nothing pending")
    return ReviewTaskOut(
        review_item_id=task.review_item_id,
        reason=str(task.reason),
        case_id=task.case_id,
        question_prompt=task.question_prompt,
        rubric_lines=list(task.rubric_lines),
        student_response=task.student_response,
        expected_awards=list(task.expected_awards),
        expected_points=task.expected_points,
        grader_awards=list(task.grader_awards),
        grader_points=task.grader_points,
        grader_feedback=task.grader_feedback,
        must_mention=list(task.must_mention),
        must_not_mention=list(task.must_not_mention),
        parsed_ok=task.parsed_ok,
    )


@router.get("/review/counts", response_model=dict[str, int])
async def counts(session: SessionDep, run_id: UUID | None = None) -> dict[str, int]:
    return await review_counts(session, run_id=run_id)


@router.post("/review/{item_id}/label", status_code=status.HTTP_204_NO_CONTENT)
async def label(item_id: UUID, payload: LabelIn, session: SessionDep) -> None:
    try:
        verdict = JudgeVerdict(payload.verdict)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"verdict must be one of {[str(v) for v in JudgeVerdict]}",
        ) from exc
    try:
        await save_label(
            session,
            review_item_id=item_id,
            labeller=payload.labeller,
            verdict=verdict,
            criteria=payload.criteria,
            notes=payload.notes,
            seconds_spent=payload.seconds_spent,
        )
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("/review/{item_id}/skip", status_code=status.HTTP_204_NO_CONTENT)
async def skip(item_id: UUID, session: SessionDep) -> None:
    try:
        await skip_review_item(session, item_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/review/pending", response_model=int)
async def pending(session: SessionDep, run_id: UUID | None = None) -> int:
    return await count_items(session, run_id=run_id)