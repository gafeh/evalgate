from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from evalgate.core.aggregate import RunSummary
from evalgate.core.audit import Verification
from evalgate.core.calibration import CalibrationReport
from evalgate.core.hashing import content_hash
from evalgate.core.regression import Comparison
from evalgate.judge.cache import RunUsage


def build_report(
    *,
    run_id: str,
    label: str,
    model: str,
    status: str,
    dataset: str,
    dataset_content_hash: str,
    summary: RunSummary,
    usage: RunUsage,
    slices: dict[str, RunSummary],
    verification: Verification,
    comparison: Comparison | None = None,
    calibration: CalibrationReport | None = None,
) -> dict[str, Any]:
    """Assemble a self-describing report and fingerprint it.

    The chain root and the report's own hash are both included, so a report
    file can be checked against the database it came from and against itself.
    """
    body: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run": {
            "id": run_id,
            "label": label,
            "model": model,
            "status": status,
            "dataset": dataset,
            "dataset_content_hash": dataset_content_hash,
        },
        "summary": {
            "cases": summary.cases,
            "unparseable": summary.unparseable,
            "mean_overall": summary.mean_overall,
            "exact_match_rate": summary.exact_match_rate,
            "over_credit_points": summary.over_credit_points,
            "under_credit_points": summary.under_credit_points,
        },
        "usage": {
            "calls": usage.calls,
            "cache_hits": usage.cache_hits,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost_usd,
        },
        "slices": {
            key: {
                "cases": value.cases,
                "mean_overall": value.mean_overall,
                "over_credit_points": value.over_credit_points,
            }
            for key, value in slices.items()
        },
        "audit": {
            "links": verification.links,
            "root": verification.root,
            "ok": verification.ok,
            "problems": [str(problem) for problem in verification.problems],
        },
    }

    if comparison is not None:
        body["gate"] = {
            "verdict": str(comparison.verdict),
            "reason": comparison.reason,
            "passed": comparison.passed,
            "paired_cases": comparison.paired_cases,
            "overall_delta": (
                None
                if comparison.overall_delta is None
                else {
                    "point": comparison.overall_delta.point,
                    "low": comparison.overall_delta.low,
                    "high": comparison.overall_delta.high,
                }
            ),
            "over_credit_delta": (
                None
                if comparison.over_credit_delta is None
                else {
                    "point": comparison.over_credit_delta.point,
                    "low": comparison.over_credit_delta.low,
                    "high": comparison.over_credit_delta.high,
                }
            ),
            "cases_needed": comparison.cases_needed,
        }

    if calibration is not None:
        body["calibration"] = {
            "pairs": calibration.pairs,
            "distinct_cases": calibration.distinct_cases,
            "agreement": calibration.agreement,
            "kappa": calibration.kappa,
            "brier": calibration.brier,
            "expected_calibration_error": calibration.expected_calibration_error,
            "trustworthy": calibration.trustworthy,
            "warnings": list(calibration.warnings),
            "recommended_threshold": calibration.recommended_threshold,
            "threshold_warnings": list(calibration.threshold_warnings),
        }

    body["report_hash"] = content_hash(body)
    return body