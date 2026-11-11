import pytest

from evalgate.core.calibration import (
    LabelPair,
    agreement_rate,
    brier_score,
    cohens_kappa,
    compute_calibration,
    expected_calibration_error,
    recommend_threshold,
    reliability_bins,
)
from evalgate.core.judgment import JudgeVerdict

PASS = JudgeVerdict.PASS
FAIL = JudgeVerdict.FAIL
PARTIAL = JudgeVerdict.PARTIAL


def pair(
    index: int, judge: JudgeVerdict, human: JudgeVerdict, confidence: float
) -> LabelPair:
    return LabelPair(
        case_id=f"c{index}",
        judge_verdict=judge,
        judge_confidence=confidence,
        human_verdict=human,
    )


def test_empty_report_is_all_zero() -> None:
    result = compute_calibration([])
    assert result.pairs == 0
    assert result.recommended_threshold is None
    assert result.bins == ()


def test_perfect_agreement() -> None:
    pairs = [pair(i, PASS, PASS, 0.9) for i in range(10)]
    assert agreement_rate(pairs) == 1.0
    assert brier_score(pairs) == pytest.approx(0.01)


def test_total_disagreement() -> None:
    pairs = [pair(i, PASS, FAIL, 0.9) for i in range(10)]
    assert agreement_rate(pairs) == 0.0
    assert brier_score(pairs) == pytest.approx(0.81)


def test_kappa_is_zero_when_a_judge_is_no_better_than_its_own_marginals() -> None:
    """A judge that always says pass on an all-pass set gets no credit."""
    pairs = [pair(i, PASS, PASS, 0.9) for i in range(10)]
    assert agreement_rate(pairs) == 1.0
    assert cohens_kappa(pairs) == 1.0

    mixed = [pair(i, PASS, PASS, 0.9) for i in range(8)]
    mixed += [pair(i + 8, PASS, FAIL, 0.9) for i in range(2)]
    assert agreement_rate(mixed) == 0.8
    # Judge said pass every time, humans said pass 80% of the time, so
    # chance agreement is already 0.8 and kappa collapses to zero.
    assert cohens_kappa(mixed) == pytest.approx(0.0)


def test_kappa_rewards_tracking_a_mixed_distribution() -> None:
    pairs = [pair(i, PASS, PASS, 0.9) for i in range(5)]
    pairs += [pair(i + 5, FAIL, FAIL, 0.9) for i in range(5)]
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_brier_rewards_honest_uncertainty() -> None:
    honest = [pair(i, PASS, PASS, 0.5) for i in range(5)]
    honest += [pair(i + 5, PASS, FAIL, 0.5) for i in range(5)]
    overconfident = [pair(i, PASS, PASS, 1.0) for i in range(5)]
    overconfident += [pair(i + 5, PASS, FAIL, 1.0) for i in range(5)]
    assert brier_score(honest) < brier_score(overconfident)


def test_bins_split_by_confidence_and_report_the_gap() -> None:
    pairs = [pair(i, PASS, PASS, 0.95) for i in range(4)]
    pairs += [pair(i + 4, PASS, FAIL, 0.15) for i in range(4)]
    bins = reliability_bins(pairs, bins=5)

    assert len(bins) == 2
    low, high = bins[0], bins[-1]
    assert low.count == 4
    assert low.accuracy == 0.0
    assert low.gap < 0.0
    assert high.accuracy == 1.0
    assert high.gap == pytest.approx(0.05)


def test_ece_is_zero_for_a_perfectly_calibrated_judge() -> None:
    pairs = [pair(i, PASS, PASS, 1.0) for i in range(5)]
    pairs += [pair(i + 5, PASS, FAIL, 0.0) for i in range(5)]
    assert expected_calibration_error(pairs, bins=5) == pytest.approx(0.0)


def test_ece_catches_overconfidence() -> None:
    pairs = [pair(i, PASS, PASS, 0.9) for i in range(5)]
    pairs += [pair(i + 5, PASS, FAIL, 0.9) for i in range(5)]
    assert expected_calibration_error(pairs, bins=5) == pytest.approx(0.4)


def test_threshold_finds_the_lowest_usable_cut() -> None:
    pairs = [pair(i, PASS, PASS, 0.9) for i in range(10)]
    pairs += [pair(i + 10, PASS, FAIL, 0.4) for i in range(10)]
    threshold = recommend_threshold(pairs, target_accuracy=0.9, min_support=5)
    assert threshold == 0.9


def test_threshold_is_none_when_the_judge_is_never_accurate_enough() -> None:
    pairs = [pair(i, PASS, FAIL, 0.95) for i in range(10)]
    assert recommend_threshold(pairs, target_accuracy=0.9) is None


def test_threshold_respects_minimum_support() -> None:
    pairs = [pair(0, PASS, PASS, 0.99)]
    pairs += [pair(i + 1, PASS, FAIL, 0.5) for i in range(10)]
    assert recommend_threshold(pairs, target_accuracy=0.9, min_support=5) is None


def test_full_report_flags_overconfidence_and_reports_coverage() -> None:
    pairs = [pair(i, PASS, PASS, 0.95) for i in range(8)]
    pairs += [pair(i + 8, PASS, FAIL, 0.6) for i in range(8)]
    result = compute_calibration(pairs, bins=5, target_accuracy=0.9, min_support=5)

    assert result.pairs == 16
    assert result.agreement == 0.5
    assert result.overconfident is True
    assert result.recommended_threshold == 0.95
    assert result.coverage_at_threshold == 0.5
    assert result.accuracy_at_threshold == 1.0


def test_bins_rejects_a_nonsense_bin_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        reliability_bins([pair(0, PASS, PASS, 0.5)], bins=0)