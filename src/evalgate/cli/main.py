from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer

from evalgate.analysis.load import load_label_pairs, load_outcomes
from evalgate.config import get_settings
from evalgate.core.aggregate import summarize, summarize_by_slice
from evalgate.core.audit import Verification
from evalgate.core.calibration import compute_calibration
from evalgate.core.regression import compare_runs
from evalgate.datasets.loader import CorpusError, load_corpus
from evalgate.db.audit import AlreadySealedError, seal_run, verify_run
from evalgate.db.repositories import (
    create_run,
    get_dataset,
    get_run,
    list_datasets,
    list_runs,
    save_dataset,
    work_item_counts,
)
from evalgate.db.session import build_engine, session_scope
from evalgate.judge.assessor import LlmJudge
from evalgate.judge.cache import run_usage
from evalgate.judge.client import AnthropicClient
from evalgate.labeller.session import REASON_SETS, enqueue_review, run_labelling
from evalgate.reporting.report import build_report
from evalgate.reporting.text import calibration_lines, comparison_lines, run_lines
from evalgate.runner.llm_grader import LlmGrader
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="LLM evaluation harness with judge calibration and regression gating.",
)
dataset_app = typer.Typer(no_args_is_help=True, help="Load and inspect datasets.")
run_app = typer.Typer(no_args_is_help=True, help="Create, inspect and compare runs.")
review_app = typer.Typer(no_args_is_help=True, help="Human review and labelling.")

app.add_typer(dataset_app, name="dataset")
app.add_typer(run_app, name="run")
app.add_typer(review_app, name="review")


def _split_spec(spec: str) -> tuple[str, str]:
    """Parse name@version."""
    if "@" not in spec:
        raise typer.BadParameter("expected name@version, for example apcs-frq@1")
    name, _, version = spec.partition("@")
    if not name or not version:
        raise typer.BadParameter("expected name@version, for example apcs-frq@1")
    return name, version


def _emit(lines: list[str]) -> None:
    for line in lines:
        typer.echo(line)


@dataset_app.command("load")
def dataset_load(
    path: Annotated[Path, typer.Argument(help="corpus directory or task file")] = Path(
        "datasets/apcs"
    ),
    name: Annotated[str, typer.Option(help="dataset name")] = "apcs-frq",
    version: Annotated[str, typer.Option(help="dataset version")] = "1",
) -> None:
    """Read a YAML corpus and store it. Idempotent for identical content."""

    async def go() -> None:
        try:
            parsed = load_corpus(path, name=name, version=version)
        except CorpusError as exc:
            raise typer.BadParameter(str(exc)) from exc

        engine = build_engine()
        async with session_scope(engine) as session:
            row = await save_dataset(session, parsed)
        await engine.dispose()
        typer.echo(f"{row.name}@{row.version}  {row.case_count} cases")
        typer.echo(f"content hash {row.content_hash}")

    asyncio.run(go())


@dataset_app.command("list")
def dataset_list() -> None:
    """List stored datasets, newest first."""

    async def go() -> None:
        engine = build_engine()
        async with session_scope(engine) as session:
            rows = await list_datasets(session)
        await engine.dispose()
        if not rows:
            typer.echo("no datasets stored")
            return
        for row in rows:
            spec = f"{row.name}@{row.version}"
            typer.echo(
                f"{spec:<34} {row.case_count:>4} cases  {row.content_hash[:12]}"
            )

    asyncio.run(go())


@run_app.command("start")
def run_start(
    dataset: Annotated[str, typer.Option(help="name@version")] = "apcs-frq@1",
    model: Annotated[str, typer.Option()] = "claude-haiku-5-5",
    label: Annotated[str, typer.Option()] = "",
) -> None:
    """Create a run and enqueue its work. Grading happens separately."""
    name, version = _split_spec(dataset)

    async def go() -> None:
        engine = build_engine()
        async with session_scope(engine) as session:
            row = await get_dataset(session, name, version)
            if row is None:
                raise typer.BadParameter(f"no dataset {name}@{version}")
            run = await create_run(
                session, dataset=row, model=model, label=label or None
            )
        await engine.dispose()
        typer.echo(str(run.id))

    asyncio.run(go())


@run_app.command("list")
def run_list() -> None:
    """List runs, newest first."""

    async def go() -> None:
        engine = build_engine()
        async with session_scope(engine) as session:
            rows = await list_runs(session)
        await engine.dispose()
        for row in rows:
            typer.echo(
                f"{row.id}  {row.status!s:<10} {row.model:<22} "
                f"{row.label or '(unlabelled)'}"
            )

    asyncio.run(go())


@run_app.command("status")
def run_status(run_id: Annotated[str, typer.Argument()]) -> None:
    """Scores, credit errors, slices and spend for one run."""
    identifier = UUID(run_id)

    async def go() -> None:
        engine = build_engine()
        async with session_scope(engine) as session:
            run = await get_run(session, identifier)
            if run is None:
                raise typer.BadParameter(f"no run {identifier}")
            outcomes = await load_outcomes(session, identifier)
            _emit(
                run_lines(
                    label=run.label or "",
                    model=run.model,
                    status=str(run.status),
                    counts=await work_item_counts(session, identifier),
                    summary=summarize(outcomes),
                    usage=await run_usage(session, identifier),
                    slices=summarize_by_slice(outcomes, "error_mode"),
                )
            )
        await engine.dispose()

    asyncio.run(go())


@run_app.command("compare")
def run_compare(
    candidate: Annotated[str, typer.Argument(help="the run under test")],
    baseline: Annotated[str, typer.Argument(help="the run to beat")],
) -> None:
    """Gate the candidate against the baseline. Exits non-zero on a regression."""

    async def go() -> None:
        engine = build_engine()
        async with session_scope(engine) as session:
            new = await load_outcomes(session, UUID(candidate))
            old = await load_outcomes(session, UUID(baseline))
        await engine.dispose()

        result = compare_runs(old, new)
        _emit(comparison_lines(result))
        if result.paired_cases == 0:
            typer.echo(
                "hint: these runs graded different datasets, so there is nothing "
                "to pair. Compare two runs over the same dataset version."
            )
        if not result.passed:
            raise typer.Exit(code=1)

    asyncio.run(go())


@app.command("grade")
def grade(
    run_id: Annotated[str, typer.Argument()],
    grader_model: Annotated[str, typer.Option()] = "claude-haiku-5-5",
    judge_model: Annotated[str, typer.Option()] = "claude-sonnet-5",
    no_judge: Annotated[bool, typer.Option("--no-judge")] = False,
    judge_auto_tool: Annotated[bool, typer.Option("--judge-auto-tool")] = False,
    batch_size: Annotated[int, typer.Option()] = 3,
) -> None:
    """Grade a run with the real model, judging each result as it goes."""
    identifier = UUID(run_id)

    async def go() -> None:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise typer.BadParameter("set ANTHROPIC_API_KEY in .env first")

        engine = build_engine()
        client = AnthropicClient(api_key=settings.anthropic_api_key, max_concurrency=3)
        grader = LlmGrader(
            client=client, engine=engine, run_id=identifier, model=grader_model
        )
        judge = (
            None
            if no_judge
            else LlmJudge(
                client=client,
                engine=engine,
                run_id=identifier,
                model=judge_model,
                force_tool=not judge_auto_tool,
            )
        )
        try:
            stats = await worker_loop(
                engine,
                run_id=identifier,
                grader=grader,
                worker_id="cli",
                batch_size=batch_size,
                judge=judge,
            )
        finally:
            await client.aclose()

        async with session_scope(engine) as session:
            usage = await run_usage(session, identifier)
        await engine.dispose()

        if stats.claimed == 0:
            typer.echo("nothing to grade, the queue is empty for this run")
        else:
            typer.echo(
                f"graded {stats.succeeded}, judged {stats.judged}, "
                f"judge failures {stats.judge_failures}, duplicates {stats.duplicates}"
            )
        typer.echo(
            f"run total: {usage.calls} calls ({usage.cache_hits} cached), "
            f"${usage.cost_usd:.4f}"
        )
        for problem in stats.errors[:5]:
            typer.echo(f"error: {problem}")

    asyncio.run(go())


@app.command("worker")
def worker(
    run_id: Annotated[str, typer.Argument()],
    worker_id: Annotated[str, typer.Option()] = "w1",
    batch_size: Annotated[int, typer.Option()] = 8,
    latency_ms: Annotated[int, typer.Option(help="simulated grader latency")] = 0,
    over_credit_rate: Annotated[float, typer.Option()] = 0.0,
    malformed_rate: Annotated[float, typer.Option()] = 0.0,
    stale_after: Annotated[float, typer.Option()] = 30.0,
) -> None:
    """Run a stub worker. For exercising the queue without spending anything."""
    identifier = UUID(run_id)

    async def go() -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        engine = build_engine()
        try:
            stats = await worker_loop(
                engine,
                run_id=identifier,
                grader=StubGrader(
                    latency_ms=latency_ms,
                    over_credit_rate=over_credit_rate,
                    malformed_rate=malformed_rate,
                ),
                worker_id=worker_id,
                batch_size=batch_size,
                stale_after_seconds=stale_after,
            )
        finally:
            await engine.dispose()
        typer.echo(
            f"claimed {stats.claimed}, succeeded {stats.succeeded}, "
            f"failed {stats.failed}, reclaimed {stats.reclaimed}"
        )

    asyncio.run(go())


@review_app.command("enqueue")
def review_enqueue(
    run_id: Annotated[str, typer.Argument()],
    sample: Annotated[int, typer.Option(help="target calibration sample size")] = 50,
    seed: Annotated[int, typer.Option()] = 0,
    no_triage: Annotated[bool, typer.Option("--no-triage")] = False,
) -> None:
    """Draw a random calibration sample, then triage anything that looks wrong."""
    asyncio.run(
        enqueue_review(UUID(run_id), sample=sample, triage=not no_triage, seed=seed)
    )


@review_app.command("label")
def review_label(
    run_id: Annotated[str, typer.Option()] = "",
    labeller: Annotated[str, typer.Option()] = "success",
    limit: Annotated[int, typer.Option()] = 25,
    reason: Annotated[str, typer.Option()] = "calibration",
) -> None:
    """Label grader feedback, blind to the judge's verdict."""
    if reason not in REASON_SETS:
        raise typer.BadParameter(f"reason must be one of {sorted(REASON_SETS)}")
    try:
        asyncio.run(
            run_labelling(
                UUID(run_id) if run_id else None,
                labeller=labeller,
                limit=limit,
                reason=reason,
            )
        )
    except KeyboardInterrupt:
        typer.echo("\ninterrupted")


@app.command("calibrate")
def calibrate(
    run_id: Annotated[str, typer.Option()] = "",
    bins: Annotated[int, typer.Option()] = 5,
    target: Annotated[
        float, typer.Option(help="accuracy the threshold must hit")
    ] = 0.90,
    include_triage: Annotated[
        bool, typer.Option("--include-triage", help="biases the estimate")
    ] = False,
) -> None:
    """Measure the judge against human labels."""

    async def go() -> None:
        engine = build_engine()
        async with session_scope(engine) as session:
            pairs = await load_label_pairs(
                session,
                run_id=UUID(run_id) if run_id else None,
                calibration_only=not include_triage,
            )
        await engine.dispose()
        report = compute_calibration(pairs, bins=bins, target_accuracy=target)
        _emit(calibration_lines(report, target=target))

    asyncio.run(go())


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Serve the HTTP API."""
    import uvicorn

    uvicorn.run("evalgate.api.app:app", host=host, port=port, reload=reload)

def _echo_verification(verification: Verification) -> None:
    typer.echo(f"links: {verification.links}")
    typer.echo(f"root:  {verification.root or '(empty chain)'}")
    if verification.ok:
        typer.echo("integrity: OK")
        return
    typer.echo("integrity: FAILED")
    for problem in verification.problems:
        typer.echo(f"  {problem}")


@app.command("seal")
def seal(
    run_id: Annotated[str, typer.Argument()],
    reseal: Annotated[bool, typer.Option("--reseal", help="replace an existing chain")] = False,
) -> None:
    """Hash-chain a finished run's results."""
    identifier = UUID(run_id)

    async def go() -> None:
        engine = build_engine()
        async with session_scope(engine) as session:
            try:
                verification = await seal_run(session, identifier, reseal=reseal)
            except AlreadySealedError as exc:
                raise typer.BadParameter(str(exc)) from exc
        await engine.dispose()
        _echo_verification(verification)
        if not verification.ok:
            raise typer.Exit(code=1)

    asyncio.run(go())


@app.command("verify")
def verify(run_id: Annotated[str, typer.Argument()]) -> None:
    """Check a sealed run against its chain. Exits non-zero on any tampering."""
    identifier = UUID(run_id)

    async def go() -> None:
        engine = build_engine()
        async with session_scope(engine) as session:
            verification = await verify_run(session, identifier)
        await engine.dispose()
        _echo_verification(verification)
        if not verification.ok:
            raise typer.Exit(code=1)

    asyncio.run(go())


@app.command("report")
def report(
    run_id: Annotated[str, typer.Argument()],
    baseline: Annotated[str, typer.Option(help="run to gate against")] = "",
    out: Annotated[Path, typer.Option(help="where to write the JSON")] = Path(
        "report.json"
    ),
    calibration: Annotated[bool, typer.Option("--calibration")] = False,
) -> None:
    """Write an auditable JSON report, optionally including a gate verdict."""
    identifier = UUID(run_id)

    async def go() -> None:
        engine = build_engine()
        async with session_scope(engine) as session:
            run = await get_run(session, identifier)
            if run is None:
                raise typer.BadParameter(f"no run {identifier}")

            outcomes = await load_outcomes(session, identifier)
            comparison = None
            if baseline:
                comparison = compare_runs(
                    await load_outcomes(session, UUID(baseline)), outcomes
                )

            calibration_report = None
            if calibration:
                calibration_report = compute_calibration(
                    await load_label_pairs(session, run_id=identifier)
                )

            body = build_report(
                run_id=str(run.id),
                label=run.label or "",
                model=run.model,
                status=str(run.status),
                dataset=str(run.dataset_id),
                dataset_content_hash=run.dataset_content_hash,
                summary=summarize(outcomes),
                usage=await run_usage(session, identifier),
                slices=summarize_by_slice(outcomes, "error_mode"),
                verification=await verify_run(session, identifier),
                comparison=comparison,
                calibration=calibration_report,
            )
        await engine.dispose()

        out.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
        typer.echo(f"wrote {out}")
        typer.echo(f"report hash {body['report_hash']}")
        typer.echo(f"audit root  {body['audit']['root']}")
        if comparison is not None:
            typer.echo(f"gate: {'PASS' if comparison.passed else 'FAIL'}")
            if not comparison.passed:
                raise typer.Exit(code=1)

    asyncio.run(go())