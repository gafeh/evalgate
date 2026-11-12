import json

from evalgate.core.aggregate import RunSummary
from evalgate.core.audit import ChainProblem, Verification, build_chain, verify_chain
from evalgate.core.calibration import compute_calibration
from evalgate.core.outcome import CaseOutcome
from evalgate.core.regression import GateConfig, compare_runs
from evalgate.judge.cache import RunUsage
from evalgate.reporting.report import build_report

FAST = GateConfig(iterations=300, seed=1)


def outcomes(scores: list[float], over: float = 0.0) -> list[CaseOutcome]:
    return [
        CaseOutcome(
            case_id=f"c{index:03d}",
            parsed_ok=True,
            overall=score,
            over_credit_points=over,
            tags={"error_mode": "syntax"},
        )
        for index, score in enumerate(scores)
    ]


def a_report(**overrides: object) -> dict[str, object]:
    base = {
        "run_id": "11111111-1111-1111-1111-111111111111",
        "label": "candidate",
        "model": "claude-haiku-5-5",
        "status": "completed",
        "dataset": "22222222-2222-2222-2222-222222222222",
        "dataset_content_hash": "a" * 64,
        "summary": RunSummary(cases=60, mean_overall=0.7975, over_credit_points=39.0),
        "usage": RunUsage(calls=120, cost_usd=0.7892),
        "slices": {"syntax": RunSummary(cases=8, mean_overall=0.0)},
        "verification": verify_chain(build_chain([{"c": 1}, {"c": 2}])),
    }
    return build_report(**{**base, **overrides})  # type: ignore[arg-type]


def test_a_report_is_json_serialisable() -> None:
    body = a_report()
    assert json.loads(json.dumps(body)) == body


def test_a_report_carries_the_audit_root() -> None:
    body = a_report()
    assert body["audit"]["ok"] is True
    assert len(body["audit"]["root"]) == 64


def test_a_report_without_a_gate_or_calibration_omits_them() -> None:
    body = a_report()
    assert "gate" not in body
    assert "calibration" not in body


def test_the_report_hash_covers_the_contents() -> None:
    first = a_report()
    second = a_report(label="other")
    assert len(first["report_hash"]) == 64
    assert first["report_hash"] != second["report_hash"]


def test_a_failing_chain_shows_up_in_the_report() -> None:
    broken = Verification(links=2, root="b" * 64, problems=())
    body = a_report(verification=broken)
    assert body["audit"]["links"] == 2

    

    worse = Verification(
        links=2, root="b" * 64, problems=(ChainProblem(1, "row", "edited"),)
    )
    body = a_report(verification=worse)
    assert body["audit"]["ok"] is False
    assert body["audit"]["problems"] == ["seq 1: row: edited"]


def test_a_gate_verdict_is_included_when_given() -> None:
    comparison = compare_runs(
        outcomes([0.9] * 40), outcomes([0.7] * 40, over=1.0), config=FAST
    )
    body = a_report(comparison=comparison)
    assert body["gate"]["verdict"] == "regressed"
    assert body["gate"]["passed"] is False
    assert body["gate"]["over_credit_delta"]["low"] > 0


def test_calibration_is_included_when_given() -> None:
    body = a_report(calibration=compute_calibration([]))
    assert body["calibration"]["pairs"] == 0
    assert body["calibration"]["trustworthy"] is False