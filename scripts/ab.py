from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.analysis.load import load_outcomes
from evalgate.core.regression import Comparison, GateConfig, compare_runs
from evalgate.datasets.synthetic import synthetic_dataset
from evalgate.db.models import Dataset
from evalgate.db.repositories import create_run, save_dataset
from evalgate.db.session import build_engine, session_scope
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader


async def _run_stub(
    engine: AsyncEngine, dataset: Dataset, label: str, grader: StubGrader
) -> list:
    async with session_scope(engine) as session:
        run = await create_run(
            session, dataset=dataset, model=grader.model_name, label=label
        )
    await worker_loop(engine, run_id=run.id, grader=grader, worker_id=label)
    async with session_scope(engine) as session:
        return await load_outcomes(session, run.id)


def report(comparison: Comparison) -> None:
    base = comparison.baseline
    cand = comparison.candidate
    print(f"paired cases: {comparison.paired_cases}")
    print(
        f"baseline  mean {base.mean_overall:.4f}  over-credit {base.mean_over_credit:.3f}"
        f"/case  unparseable {base.unparseable}"
    )
    print(
        f"candidate mean {cand.mean_overall:.4f}  over-credit {cand.mean_over_credit:.3f}"
        f"/case  unparseable {cand.unparseable}"
    )
    if comparison.overall_delta is not None:
        print(f"score delta       {comparison.overall_delta}")
    if comparison.over_credit_delta is not None:
        print(f"over-credit delta {comparison.over_credit_delta}")
    print()
    for item in comparison.slices:
        print(
            f"  {item.tag_value:<14} n={item.cases:<4} "
            f"{item.baseline_mean:.3f} -> {item.candidate_mean:.3f} "
            f"({item.delta:+.3f})"
        )
    print()
    print(f"verdict: {comparison.verdict.upper()}")
    print(f"reason:  {comparison.reason}")
    if comparison.cases_needed is not None:
        print(f"would need about {comparison.cases_needed} paired cases to decide")
    print(f"gate: {'PASS' if comparison.passed else 'FAIL'}")


async def main(size: int, version: str, over_credit: float, malformed: float) -> None:
    logging.basicConfig(level=logging.WARNING)
    engine = build_engine()

    async with session_scope(engine) as session:
        dataset = await save_dataset(session, synthetic_dataset(size, version))

    baseline = await _run_stub(engine, dataset, "baseline", StubGrader())
    candidate = await _run_stub(
        engine,
        dataset,
        "candidate",
        StubGrader(
            model_name="stub-v2",
            over_credit_rate=over_credit,
            malformed_rate=malformed,
        ),
    )
    await engine.dispose()

    report(compare_runs(baseline, candidate, config=GateConfig()))


def cli() -> None:
    parser = argparse.ArgumentParser(prog="ab")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--version", default="ab-1")
    parser.add_argument("--over-credit", type=float, default=0.0)
    parser.add_argument("--malformed", type=float, default=0.0)
    args = parser.parse_args()
    asyncio.run(main(args.size, args.version, args.over_credit, args.malformed))


if __name__ == "__main__":
    cli()