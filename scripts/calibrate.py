from __future__ import annotations

import argparse
import asyncio
from uuid import UUID

from evalgate.analysis.load import load_label_pairs
from evalgate.core.calibration import CalibrationReport, compute_calibration
from evalgate.db.session import build_engine, session_scope


def report(result: CalibrationReport, target: float) -> None:
    if result.pairs == 0:
        print("no labelled calibration pairs yet")
        return

    if result.warnings:
        print("DO NOT QUOTE THESE NUMBERS AS A MEASUREMENT")
        for note in result.warnings:
            print(f"  - {note}")
        print()

    print(f"labelled calibration pairs: {result.pairs}")
    print(f"distinct cases:             {result.distinct_cases}")
    print(f"human agreement:            {result.agreement:.3f}")
    print(f"cohen's kappa:              {result.kappa:.3f}")
    print(f"brier score:                {result.brier:.4f}  (lower is better)")
    print(f"expected calibration error: {result.expected_calibration_error:.4f}")
    print(f"mean judge confidence:      {result.mean_confidence:.3f}")
    print(
        "the judge is "
        + ("OVERCONFIDENT" if result.overconfident else "not overconfident")
    )
    print()

    print("reliability curve")
    print(f"  {'bin':<14}{'n':>5}{'claimed':>10}{'actual':>9}{'gap':>9}")
    for item in result.bins:
        label = f"{item.low:.1f}-{item.high:.1f}"
        print(
            f"  {label:<14}{item.count:>5}{item.mean_confidence:>10.3f}"
            f"{item.accuracy:>9.3f}{item.gap:>+9.3f}"
        )
    print()

    print("agreement by judge verdict")
    for verdict, rate in sorted(result.agreement_by_judge_verdict.items()):
        print(f"  {verdict:<10}{rate:.3f}")
    print()

    if result.recommended_threshold is None:
        print(
            f"no confidence cut reaches {target:.0%} accuracy with enough support. "
            "Every result should go to review until the judge improves or more "
            "labels arrive."
        )
    else:
        print(f"recommended review threshold: {result.recommended_threshold:.3f}")
        print(
            f"  auto-approving above it covers {result.coverage_at_threshold:.1%} "
            f"of cases at {result.accuracy_at_threshold:.1%} accuracy"
        )

    if result.warnings:
        print()
        print("reminder: the warnings above apply to everything printed here")


async def main(
    run_id: UUID | None, bins: int, target: float, include_triage: bool
) -> None:
    engine = build_engine()
    async with session_scope(engine) as session:
        pairs = await load_label_pairs(
            session, run_id=run_id, calibration_only=not include_triage
        )
    await engine.dispose()
    report(compute_calibration(pairs, bins=bins, target_accuracy=target), target)


def cli() -> None:
    parser = argparse.ArgumentParser(prog="calibrate")
    parser.add_argument("--run-id")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument(
        "--include-triage",
        action="store_true",
        help="include triaged items, which biases the estimate",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            UUID(args.run_id) if args.run_id else None,
            args.bins,
            args.target,
            args.include_triage,
        )
    )


if __name__ == "__main__":
    cli()