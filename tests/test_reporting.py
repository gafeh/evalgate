from evalgate.core.aggregate import RunSummary
from evalgate.core.calibration import LabelPair, compute_calibration
from evalgate.core.judgment import JudgeVerdict
from evalgate.core.outcome import CaseOutcome
from evalgate.core.regression import GateConfig, compare_runs
from evalgate.judge.cache import RunUsage
from evalgate.reporting.text import (
    calibration_lines,
    comparison_lines,
    run_lines,
)

FAST = GateConfig(iterations=400, seed=3)


def outcomes(scores: list[float], over: float = 0.0) -> list[CaseOutcome]:
    return [
        CaseOutcome(
            case_id=f"c{index:03d}",
            parsed_ok=True,
            overall=score,
            over_credit_points=over,
            exact_match=score == 1.0,
            tags={"error_mode": "off_by_one"},
        )
        for index, score in enumerate(scores)
    ]


def test_run_lines_report_the_essentials() -> None:
    text = "\n".join(
        run_lines(
            label="baseline",
            model="claude-haiku-5-5",
            status="completed",
            counts={"done": 12},
            summary=RunSummary(
                cases=12,
                unparseable=1,
                mean_overall=0.75,
                exact_match_rate=0.5,
                over_credit_points=4.0,
                under_credit_points=0.0,
            ),
            usage=RunUsage(calls=24, cache_hits=12, cost_usd=0.0442),
            slices={"off_by_one": RunSummary(cases=12, mean_overall=0.75)},
        )
    )
    assert "baseline" in text
    assert "1 unparseable" in text
    assert "0.7500" in text
    assert "4 over" in text
    assert "$0.0442" in text
    assert "off_by_one" in text


def test_run_lines_cope_with_an_empty_run() -> None:
    text = "\n".join(
        run_lines(
            label="",
            model="stub-v1",
            status="pending",
            counts={},
            summary=RunSummary(),
            usage=RunUsage(),
            slices={},
        )
    )
    assert "(unlabelled)" in text
    assert "(empty)" in text


def test_comparison_lines_show_the_verdict_and_gate() -> None:
    base = outcomes([0.9] * 40)
    cand = outcomes([0.75] * 40, over=1.0)
    text = "\n".join(comparison_lines(compare_runs(base, cand, config=FAST)))
    assert "verdict: REGRESSED" in text
    assert "gate: FAIL" in text
    assert "over-credit delta" in text
    assert "off_by_one" in text


def test_comparison_lines_report_an_inconclusive_sample_size() -> None:
    base = outcomes([0.1, 0.9, 0.5, 0.2, 0.8, 0.4])
    cand = outcomes([0.9, 0.1, 0.4, 0.9, 0.2, 0.7])
    text = "\n".join(comparison_lines(compare_runs(base, cand, config=FAST)))
    assert "INCONCLUSIVE" in text
    assert "paired cases to decide" in text


def test_calibration_lines_lead_with_the_warnings() -> None:
    pairs = [
        LabelPair(
            case_id=f"c{index}",
            judge_verdict=JudgeVerdict.PASS,
            judge_confidence=0.9,
            human_verdict=JudgeVerdict.PASS,
        )
        for index in range(5)
    ]
    lines = calibration_lines(compute_calibration(pairs), target=0.9)
    assert lines[0] == "DO NOT QUOTE THESE NUMBERS AS A MEASUREMENT"
    assert any("labelled pairs" in line for line in lines)


def test_calibration_lines_with_nothing_labelled() -> None:
    assert calibration_lines(compute_calibration([]), target=0.9) == [
        "no labelled calibration pairs yet"
    ]


def test_calibration_lines_show_a_recommended_threshold() -> None:
    pairs = [
        LabelPair(
            case_id=f"hi{index}",
            judge_verdict=JudgeVerdict.PASS,
            judge_confidence=0.95,
            human_verdict=JudgeVerdict.PASS,
        )
        for index in range(25)
    ]
    pairs += [
        LabelPair(
            case_id=f"lo{index}",
            judge_verdict=JudgeVerdict.PASS,
            judge_confidence=0.3,
            human_verdict=JudgeVerdict.FAIL,
        )
        for index in range(15)
    ]
    text = "\n".join(calibration_lines(compute_calibration(pairs), target=0.9))
    assert "DO NOT QUOTE" not in text
    assert "recommended review threshold: 0.950" in text
    assert "62.5%" in text