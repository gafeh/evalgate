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
    assert result.warnings == ("no labelled pairs",)


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


def test_a_small_sample_is_flagged() -> None:
    result = compute_calibration([pair(i, PASS, PASS, 0.9) for i in range(5)])
    assert result.trustworthy is False
    assert any("effective sample size" in note for note in result.warnings)


def test_repeated_cases_are_flagged() -> None:
    pairs = [
        LabelPair(
            case_id="same", judge_verdict=PASS, judge_confidence=0.9, human_verdict=PASS
        )
        for _ in range(40)
    ]
    result = compute_calibration(pairs)
    assert result.distinct_cases == 1
    assert any("distinct cases" in note for note in result.warnings)


def test_a_single_confidence_bin_is_flagged() -> None:
    pairs = [pair(i, PASS, PASS, 0.9) for i in range(20)]
    pairs += [pair(i + 20, PASS, FAIL, 0.95) for i in range(20)]
    result = compute_calibration(pairs)
    assert any("one bin" in note for note in result.warnings)


def test_perfect_agreement_is_flagged() -> None:
    pairs = [pair(i, PASS, PASS, 0.5 + i * 0.01) for i in range(40)]
    result = compute_calibration(pairs)
    assert any("perfect agreement" in note for note in result.warnings)


def test_a_healthy_report_has_no_warnings() -> None:
    pairs = [pair(i, PASS, PASS, 0.95) for i in range(25)]
    pairs += [pair(i + 25, PASS, FAIL, 0.3) for i in range(15)]
    result = compute_calibration(pairs)
    assert result.warnings == ()
    assert result.trustworthy is True


def test_a_cut_that_reviews_nothing_is_flagged() -> None:
    """95% accuracy everywhere means the lowest cut wins and nobody reviews."""
    pairs = [pair(i, PASS, PASS, 0.9) for i in range(38)]
    pairs += [pair(i + 38, PASS, FAIL, 0.95) for i in range(2)]
    result = compute_calibration(pairs)

    assert result.recommended_threshold is not None
    assert result.coverage_at_threshold == 1.0
    assert result.pairs_below_threshold == 0
    assert any("nothing to review" in note for note in result.threshold_warnings)


def test_an_unmeasured_low_side_is_flagged() -> None:
    """The cut sits above the low group, but only three pairs are down there."""
    pairs = [pair(i, PASS, PASS, 0.95) for i in range(33)]
    pairs += [pair(i + 33, PASS, FAIL, 0.95) for i in range(2)]
    pairs += [pair(i + 35, PASS, FAIL, 0.2) for i in range(3)]
    result = compute_calibration(pairs)

    assert result.recommended_threshold == 0.95
    assert result.pairs_below_threshold == 3
    assert result.coverage_at_threshold < 1.0
    assert any("low side is unmeasured" in note for note in result.threshold_warnings)


def test_a_well_supported_cut_has_no_threshold_warnings() -> None:
    pairs = [pair(i, PASS, PASS, 0.95) for i in range(30)]
    pairs += [pair(i + 30, PASS, FAIL, 0.3) for i in range(20)]
    result = compute_calibration(pairs)

    assert result.recommended_threshold == 0.95
    assert result.threshold_warnings == ()
    assert result.coverage_at_threshold < 1.0
    assert result.pairs_below_threshold == 20