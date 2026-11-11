from evalgate.core.outcome import CaseOutcome
from evalgate.core.regression import (
    GateConfig,
    RegressionVerdict,
    compare_runs,
)

FAST = GateConfig(iterations=600, seed=5)


def run(
    scores: list[float],
    *,
    over: list[float] | None = None,
    modes: list[str] | None = None,
) -> list[CaseOutcome]:
    overs = over or [0.0] * len(scores)
    tags = modes or ["off_by_one"] * len(scores)
    return [
        CaseOutcome(
            case_id=f"case-{index:03d}",
            parsed_ok=True,
            overall=score,
            over_credit_points=overs[index],
            exact_match=score == 1.0,
            tags={"error_mode": tags[index]},
        )
        for index, score in enumerate(scores)
    ]


def test_identical_runs_report_no_change_and_pass() -> None:
    scores = [0.8, 0.9, 0.7, 0.85, 0.75] * 6
    result = compare_runs(run(scores), run(scores), config=FAST)
    assert result.verdict is RegressionVerdict.NO_CHANGE
    assert result.passed is True
    assert result.overall_delta is not None
    assert result.overall_delta.point == 0.0
    assert result.paired_cases == 30


def test_a_real_drop_is_caught() -> None:
    baseline = [0.9, 0.85, 0.95, 0.8, 0.9] * 8
    candidate = [score - 0.12 for score in baseline]
    result = compare_runs(run(baseline), run(candidate), config=FAST)
    assert result.verdict is RegressionVerdict.REGRESSED
    assert result.passed is False
    assert "fell" in result.reason


def test_a_real_improvement_is_recognised() -> None:
    baseline = [0.5, 0.55, 0.45, 0.6, 0.5] * 8
    candidate = [score + 0.15 for score in baseline]
    result = compare_runs(run(baseline), run(candidate), config=FAST)
    assert result.verdict is RegressionVerdict.IMPROVED
    assert result.passed is True


def test_a_tiny_difference_does_not_trip_the_gate() -> None:
    baseline = [0.8] * 40
    candidate = [0.795] * 40
    result = compare_runs(run(baseline), run(candidate), config=FAST)
    assert result.verdict is RegressionVerdict.NO_CHANGE
    assert result.passed is True


def test_noisy_small_samples_are_inconclusive_not_a_pass() -> None:
    baseline = [0.1, 0.9, 0.5, 0.2, 0.8, 0.4]
    candidate = [0.9, 0.1, 0.4, 0.9, 0.2, 0.7]
    result = compare_runs(run(baseline), run(candidate), config=FAST)
    assert result.verdict is RegressionVerdict.INCONCLUSIVE
    assert result.passed is False
    assert result.cases_needed is not None
    assert result.cases_needed > len(baseline)


def test_rising_over_credit_regresses_even_when_the_score_improves() -> None:
    baseline = run([0.7] * 30, over=[0.0] * 30)
    candidate = run([0.85] * 30, over=[1.0] * 30)
    result = compare_runs(baseline, candidate, config=FAST)
    assert result.verdict is RegressionVerdict.REGRESSED
    assert "over-credit" in result.reason


def test_slice_breakdown_isolates_where_it_broke() -> None:
    modes = ["off_by_one"] * 20 + ["logic"] * 20
    baseline = run([0.9] * 40, modes=modes)
    candidate = run([0.9] * 20 + [0.3] * 20, modes=modes)
    result = compare_runs(baseline, candidate, config=FAST)

    by_mode = {item.tag_value: item for item in result.slices}
    assert by_mode["off_by_one"].delta == 0.0
    assert by_mode["logic"].delta < -0.5
    assert result.verdict is RegressionVerdict.REGRESSED


def test_runs_with_no_shared_cases_are_inconclusive() -> None:
    baseline = run([0.8] * 5)
    candidate = [
        CaseOutcome(case_id=f"other-{i}", parsed_ok=True, overall=0.8) for i in range(5)
    ]
    result = compare_runs(baseline, candidate, config=FAST)
    assert result.verdict is RegressionVerdict.INCONCLUSIVE
    assert result.paired_cases == 0
    assert result.overall_delta is None


def test_only_shared_cases_are_compared() -> None:
    baseline = run([0.8] * 10)
    candidate = run([0.8] * 6)
    result = compare_runs(baseline, candidate, config=FAST)
    assert result.paired_cases == 6
    assert result.baseline.cases == 6
def test_a_certain_but_trivial_shift_is_no_change_not_inconclusive() -> None:
    """Zero variance and a sub-threshold effect is the most conclusive case there is."""
    result = compare_runs(run([0.8] * 40), run([0.795] * 40), config=FAST)
    assert result.overall_delta is not None
    assert result.overall_delta.width == 0.0
    assert result.overall_delta.excludes_zero is True
    assert result.verdict is RegressionVerdict.NO_CHANGE
    assert result.passed is True