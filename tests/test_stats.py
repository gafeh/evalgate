import pytest

from evalgate.core.stats import (
    bootstrap_mean_ci,
    mean,
    paired_delta_ci,
    standard_deviation,
    suggested_paired_sample_size,
    z_for_confidence,
)


def test_mean_of_empty_is_zero() -> None:
    assert mean([]) == 0.0


def test_standard_deviation_needs_two_values() -> None:
    assert standard_deviation([]) == 0.0
    assert standard_deviation([3.0]) == 0.0
    assert standard_deviation([1.0, 1.0]) == 0.0
    assert standard_deviation([0.0, 2.0]) > 0.0


def test_bootstrap_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_mean_ci([])


def test_bootstrap_of_a_constant_sample_has_zero_width() -> None:
    interval = bootstrap_mean_ci([0.5] * 20, iterations=200)
    assert interval.point == 0.5
    assert interval.width == 0.0


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    values = [0.1, 0.4, 0.5, 0.55, 0.6, 0.9, 0.3, 0.7]
    interval = bootstrap_mean_ci(values, iterations=1000, seed=7)
    assert interval.low <= interval.point <= interval.high
    assert interval.width > 0.0


def test_bootstrap_is_deterministic_for_a_given_seed() -> None:
    values = [0.2, 0.4, 0.6, 0.8]
    first = bootstrap_mean_ci(values, iterations=500, seed=42)
    second = bootstrap_mean_ci(values, iterations=500, seed=42)
    other = bootstrap_mean_ci(values, iterations=500, seed=43)
    assert first == second
    assert first.low != other.low or first.high != other.high


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8]
    narrow = bootstrap_mean_ci(values, confidence=0.80, iterations=1000, seed=3)
    wide = bootstrap_mean_ci(values, confidence=0.99, iterations=1000, seed=3)
    assert wide.width > narrow.width


def test_paired_delta_of_identical_runs_is_zero() -> None:
    values = [0.3, 0.6, 0.9, 0.1]
    interval = paired_delta_ci(values, values, iterations=200)
    assert interval.point == 0.0
    assert interval.width == 0.0
    assert interval.excludes_zero is False


def test_paired_delta_detects_a_consistent_shift() -> None:
    baseline = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.6, 0.4]
    candidate = [value - 0.1 for value in baseline]
    interval = paired_delta_ci(baseline, candidate, iterations=1000, seed=1)
    assert interval.point == pytest.approx(-0.1)
    assert interval.high < 0.0
    assert interval.excludes_zero is True


def test_paired_delta_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="match in length"):
        paired_delta_ci([0.1, 0.2], [0.1])


def test_z_grows_with_confidence() -> None:
    assert z_for_confidence(0.95) == pytest.approx(1.96, abs=0.01)
    assert z_for_confidence(0.99) > z_for_confidence(0.95)


def test_sample_size_grows_with_variance_and_shrinks_with_effect() -> None:
    assert suggested_paired_sample_size(0.0, 0.02) == 1
    noisy = suggested_paired_sample_size(0.2, 0.02)
    quiet = suggested_paired_sample_size(0.05, 0.02)
    assert noisy > quiet
    assert suggested_paired_sample_size(0.2, 0.10) < noisy


def test_sample_size_rejects_a_zero_effect() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        suggested_paired_sample_size(0.1, 0.0)