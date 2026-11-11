from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_ITERATIONS = 5_000


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float
    confidence: float

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        return f"{self.point:+.4f} [{self.low:+.4f}, {self.high:+.4f}]"


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def standard_deviation(values: Sequence[float]) -> float:
    """Sample standard deviation. Zero for fewer than two values."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile. Input must already be sorted."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for the mean.

    Seeded, so the same inputs give the same interval. An eval gate that
    returns a different verdict on reruns of identical data is useless.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    point = mean(values)
    if len(values) == 1:
        return ConfidenceInterval(point, point, point, confidence)

    rng = random.Random(seed)
    population = list(values)
    size = len(population)
    resampled = [
        statistics.fmean(rng.choices(population, k=size)) for _ in range(iterations)
    ]
    resampled.sort()

    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        point=point,
        low=_percentile(resampled, alpha),
        high=_percentile(resampled, 1.0 - alpha),
        confidence=confidence,
    )


def paired_delta_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> ConfidenceInterval:
    """Interval for the mean of candidate minus baseline, paired by position.

    Pairing removes between-case variance, which is the dominant source of
    noise here: some cases are simply harder than others, and an unpaired
    comparison spends all its statistical power rediscovering that.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired samples must match in length, got {len(baseline)} and {len(candidate)}"
        )
    deltas = [c - b for b, c in zip(baseline, candidate, strict=True)]
    return bootstrap_mean_ci(
        deltas, confidence=confidence, iterations=iterations, seed=seed
    )


def z_for_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


def suggested_paired_sample_size(
    delta_sd: float, min_effect: float, *, confidence: float = 0.95
) -> int:
    """How many paired cases would resolve an effect of min_effect.

    Rough and deliberately so: it answers "roughly how many more cases do I
    need" when a comparison comes back inconclusive.
    """
    if min_effect <= 0.0:
        raise ValueError("min_effect must be positive")
    if delta_sd <= 0.0:
        return 1
    return max(1, math.ceil((z_for_confidence(confidence) * delta_sd / min_effect) ** 2))