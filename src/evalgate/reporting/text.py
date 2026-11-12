from __future__ import annotations

from evalgate.core.aggregate import RunSummary
from evalgate.core.calibration import CalibrationReport
from evalgate.core.regression import Comparison
from evalgate.judge.cache import RunUsage


def run_lines(
    *,
    label: str,
    model: str,
    status: str,
    counts: dict[str, int],
    summary: RunSummary,
    usage: RunUsage,
    slices: dict[str, RunSummary],
) -> list[str]:
    lines = [
        f"run     {label or '(unlabelled)'}  model={model}  status={status}",
        f"queue   {counts or '(empty)'}",
        f"cases   {summary.cases} scored, {summary.unparseable} unparseable",
        f"mean    {summary.mean_overall:.4f}",
        f"exact   {summary.exact_match_rate:.1%} of cases matched the rubric exactly",
        f"credit  {summary.over_credit_points:g} over, "
        f"{summary.under_credit_points:g} under",
        f"spend   {usage.calls} calls ({usage.cache_hits} cached), "
        f"${usage.cost_usd:.4f}",
    ]
    if slices:
        lines.append("")
        lines.append("by error mode")
        for key, value in slices.items():
            lines.append(
                f"  {key:<18} n={value.cases:<4} mean={value.mean_overall:.3f}  "
                f"over={value.over_credit_points:g}"
            )
    return lines


def comparison_lines(comparison: Comparison) -> list[str]:
    base = comparison.baseline
    cand = comparison.candidate
    lines = [
        f"paired cases: {comparison.paired_cases}",
        f"baseline  mean {base.mean_overall:.4f}  "
        f"over-credit {base.mean_over_credit:.3f}/case  "
        f"unparseable {base.unparseable}",
        f"candidate mean {cand.mean_overall:.4f}  "
        f"over-credit {cand.mean_over_credit:.3f}/case  "
        f"unparseable {cand.unparseable}",
    ]
    if comparison.overall_delta is not None:
        lines.append(f"score delta       {comparison.overall_delta}")
    if comparison.over_credit_delta is not None:
        lines.append(f"over-credit delta {comparison.over_credit_delta}")

    if comparison.slices:
        lines.append("")
        for item in comparison.slices:
            lines.append(
                f"  {item.tag_value:<18} n={item.cases:<4} "
                f"{item.baseline_mean:.3f} -> {item.candidate_mean:.3f} "
                f"({item.delta:+.3f})"
            )

    lines.extend(
        [
            "",
            f"verdict: {str(comparison.verdict).upper()}",
            f"reason:  {comparison.reason}",
        ]
    )
    if comparison.cases_needed is not None:
        lines.append(
            f"would need about {comparison.cases_needed} paired cases to decide"
        )
    lines.append(f"gate: {'PASS' if comparison.passed else 'FAIL'}")
    return lines


def calibration_lines(report: CalibrationReport, *, target: float) -> list[str]:
    if report.pairs == 0:
        return ["no labelled calibration pairs yet"]

    lines: list[str] = []
    if report.warnings:
        lines.append("DO NOT QUOTE THESE NUMBERS AS A MEASUREMENT")
        lines.extend(f"  - {note}" for note in report.warnings)
        lines.append("")

    lines.extend(
        [
            f"labelled calibration pairs: {report.pairs}",
            f"effective sample size:      {report.effective_pairs:.1f}"
            + ("  (weighted)" if report.weighted else ""),
            f"distinct cases:             {report.distinct_cases}",
            f"human agreement:            {report.agreement:.3f}",
            f"cohen's kappa:              {report.kappa:.3f}",
            f"brier score:                {report.brier:.4f}  (lower is better)",
            f"expected calibration error: {report.expected_calibration_error:.4f}",
            f"mean judge confidence:      {report.mean_confidence:.3f}",
            "the judge is "
            + ("OVERCONFIDENT" if report.overconfident else "not overconfident"),
        ]
    )

    if report.strata:
        lines.append("")
        lines.append("labels per stratum")
        for name, count in report.strata.items():
            lines.append(f"  {name:<12}{count:>4}")

    lines.extend(
        [
            "",
            "reliability curve",
            f"  {'bin':<14}{'n':>5}{'claimed':>10}{'actual':>9}{'gap':>9}",
        ]
    )
    for item in report.bins:
        label = f"{item.low:.1f}-{item.high:.1f}"
        lines.append(
            f"  {label:<14}{item.count:>5}{item.mean_confidence:>10.3f}"
            f"{item.accuracy:>9.3f}{item.gap:>+9.3f}"
        )

    lines.extend(["", "agreement by judge verdict"])
    for verdict, rate in sorted(report.agreement_by_judge_verdict.items()):
        lines.append(f"  {verdict:<10}{rate:.3f}")
    lines.append("")

    if report.recommended_threshold is None:
        lines.append(
            f"no confidence cut reaches {target:.0%} accuracy with enough support, "
            "so every result should go to review"
        )
    else:
        lines.append(f"recommended review threshold: {report.recommended_threshold:.3f}")
        lines.append(
            f"  above it: {report.coverage_at_threshold:.1%} of cases at "
            f"{report.accuracy_at_threshold:.1%} accuracy"
        )
        lines.append(
            f"  below it: {report.pairs_below_threshold} pairs at "
            f"{report.accuracy_below_threshold:.1%} accuracy"
        )
        for note in report.threshold_warnings:
            lines.append(f"  ! {note}")

    if report.warnings:
        lines.extend(["", "reminder: the warnings above apply to everything here"])
    return lines