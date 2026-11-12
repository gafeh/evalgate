from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from evalgate.analysis.load import load_label_pairs, load_outcomes
from evalgate.api.deps import SessionDep
from evalgate.core.aggregate import summarize, summarize_by_slice
from evalgate.core.calibration import compute_calibration
from evalgate.core.judgment import JudgeVerdict
from evalgate.core.regression import compare_runs
from evalgate.db.models import ReviewReason, ReviewStatus
from evalgate.db.repositories import get_run, list_runs
from evalgate.db.review import (
    count_items,
    next_review_task,
    save_label,
    skip_review_item,
)
from evalgate.judge.cache import run_usage
from evalgate.web import TEMPLATES_DIR

router = APIRouter(prefix="/ui", tags=["ui"], include_in_schema=False)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

CALIBRATION = (ReviewReason.CALIBRATION,)


@router.get("", response_class=HTMLResponse)
async def runs_page(request: Request, session: SessionDep) -> HTMLResponse:
    rows = []
    for run in await list_runs(session, limit=50):
        outcomes = await load_outcomes(session, run.id)
        summary = summarize(outcomes)
        rows.append(
            {
                "id": str(run.id),
                "short": str(run.id)[:8],
                "label": run.label,
                "model": run.model,
                "status": str(run.status),
                "cases": summary.cases,
                "mean": summary.mean_overall,
                "over": summary.over_credit_points,
            }
        )
    return templates.TemplateResponse(
        request=request, name="runs.html", context={"runs": rows, "nav": "runs"}
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_page(
    request: Request, run_id: UUID, session: SessionDep
) -> HTMLResponse:
    run = await get_run(session, run_id)
    if run is None:
        return RedirectResponse("/ui", status_code=303)  # type: ignore[return-value]

    from evalgate.api.routers.runs import results as results_endpoint

    outcomes = await load_outcomes(session, run_id)
    return templates.TemplateResponse(
        request=request,
        name="run.html",
        context={
            "nav": "runs",
            "run": {
                "id": str(run.id),
                "label": run.label,
                "model": run.model,
                "status": str(run.status),
            },
            "summary": summarize(outcomes),
            "usage": await run_usage(session, run_id),
            "slices": summarize_by_slice(outcomes, "error_mode"),
            "results": await results_endpoint(run_id, session),
        },
    )


@router.get("/runs/{run_id}/compare", response_class=HTMLResponse)
async def compare_page(
    request: Request, run_id: UUID, baseline: UUID, session: SessionDep
) -> HTMLResponse:
    result = compare_runs(
        await load_outcomes(session, baseline), await load_outcomes(session, run_id)
    )
    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={
            "nav": "runs",
            "result": result,
            "candidate": str(run_id),
            "baseline": str(baseline),
        },
    )


@router.get("/calibration", response_class=HTMLResponse)
async def calibration_page(
    request: Request, session: SessionDep, run_id: str = ""
) -> HTMLResponse:
    identifier = UUID(run_id) if run_id else None
    pairs = await load_label_pairs(session, run_id=identifier)
    return templates.TemplateResponse(
        request=request,
        name="calibration.html",
        context={
            "nav": "calibration",
            "run_id": run_id,
            "report": compute_calibration(pairs),
        },
    )


async def _card_context(
    request: Request, session: SessionDep, run_id: str
) -> dict[str, object]:
    identifier = UUID(run_id) if run_id else None
    return {
        "nav": "label",
        "run_id": run_id,
        "read_only": bool(getattr(request.app.state, "read_only", False)),
        "task": await next_review_task(
            session, run_id=identifier, reasons=CALIBRATION
        ),
        "remaining": await count_items(
            session,
            run_id=identifier,
            reasons=CALIBRATION,
            status=ReviewStatus.PENDING,
        ),
    }


@router.get("/label", response_class=HTMLResponse)
async def label_page(
    request: Request, session: SessionDep, run_id: str = ""
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="label.html",
        context=await _card_context(request, session, run_id),
    )


@router.post("/label", response_class=HTMLResponse)
async def submit_label(
    request: Request,
    session: SessionDep,
    review_item_id: Annotated[UUID, Form()],
    verdict: Annotated[str, Form()],
    run_id: Annotated[str, Form()] = "",
    notes: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Save one label and swap in the next card."""
    if verdict == "skip":
        await skip_review_item(session, review_item_id)
    else:
        await save_label(
            session,
            review_item_id=review_item_id,
            labeller="web",
            verdict=JudgeVerdict(verdict),
            notes=notes.strip(),
        )
    return templates.TemplateResponse(
        request=request,
        name="_card.html",
        context=await _card_context(request, session, run_id),
    )