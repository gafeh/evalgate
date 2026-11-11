from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from evalgate.core.judgment import JudgeVerdict

DEFAULT_BINS = 5
DEFAULT_TARGET_ACCURACY = 0.90
DEFAULT_MIN_SUPPORT = 5

MIN_RELIABLE_PAIRS = 30
MIN_OCCUPIED_BINS = 2


@dataclass(frozen=True, slots=True)
class LabelPair:
    """One case where both the judge and a human gave a verdict."""

    case_id: str
    judge_verdict: JudgeVerdict
    judge_confidence: float
    human_verdict: JudgeVerdict

    @property
    def agrees(self) -> bool:
        return self.judge_verdict == self.human_verdict


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Positive means underconfident, negative means overconfident."""
        return self.accuracy - self.mean_confidence


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    pairs: int = 0
    distinct_cases: int = 0
    agreement: float = 0.0
    kappa: float = 0.0
    brier: float = 0.0
    expected_calibration_error: float = 0.0
    mean_confidence: float = 0.0
    bins: tuple[ReliabilityBin, ...] = ()
    agreement_by_judge_verdict: dict[str, float] = field(default_factory=dict)
    recommended_threshold: float | None = None
    coverage_at_threshold: float = 0.0
    accuracy_at_threshold: float = 0.0
    warnings: tuple[str, ...] = ()

    @property
    def overconfident(self) -> bool:
        return self.mean_confidence > self.agreement

    @property
    def trustworthy(self) -> bool:
        """Whether these numbers should be quoted as a measurement."""
        return not self.warnings


def agreement_rate(pairs: Sequence[LabelPair]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for pair in pairs if pair.agrees) / len(pairs)


def cohens_kappa(pairs: Sequence[LabelPair]) -> float:
    """Agreement corrected for what chance alone would produce.

    Raw agreement flatters a judge that always says "pass" on a dataset
    that is mostly passes. Kappa removes that: zero means no better than
    guessing with the same marginal frequencies.
    """
    if not pairs:
        return 0.0

    total = len(pairs)
    observed = agreement_rate(pairs)

    expected = 0.0
    for verdict in JudgeVerdict:
        judge_share = sum(1 for p in pairs if p.judge_verdict == verdict) / total
        human_share = sum(1 for p in pairs if p.human_verdict == verdict) / total
        expected += judge_share * human_share

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def brier_score(pairs: Sequence[LabelPair]) -> float:
    """Mean squared error of confidence against being right. Lower is better.

    This is the number that tests whether confidence is a probability or
    just a vibe. A judge claiming 0.9 should be wrong about one time in ten.
    """
    if not pairs:
        return 0.0
    return sum(
        (pair.judge_confidence - (1.0 if pair.agrees else 0.0)) ** 2 for pair in pairs
    ) / len(pairs)


def reliability_bins(
    pairs: Sequence[LabelPair], *, bins: int = DEFAULT_BINS
) -> tuple[ReliabilityBin, ...]:
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if not pairs:
        return ()

    width = 1.0 / bins
    buckets: list[list[LabelPair]] = [[] for _ in range(bins)]
    for pair in pairs:
        index = min(bins - 1, int(pair.judge_confidence / width))
        buckets[max(0, index)].append(pair)

    out: list[ReliabilityBin] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        out.append(
            ReliabilityBin(
                low=index * width,
                high=(index + 1) * width,
                count=len(bucket),
                mean_confidence=sum(p.judge_confidence for p in bucket) / len(bucket),
                accuracy=agreement_rate(bucket),
            )
        )
    return tuple(out)


def expected_calibration_error(
    pairs: Sequence[LabelPair], *, bins: int = DEFAULT_BINS
) -> float:
    """Weighted average gap between claimed confidence and actual accuracy."""
    if not pairs:
        return 0.0
    total = len(pairs)
    return sum(
        (item.count / total) * abs(item.accuracy - item.mean_confidence)
        for item in reliability_bins(pairs, bins=bins)
    )


def recommend_threshold(
    pairs: Sequence[LabelPair],
    *,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> float | None:
    """Lowest confidence cut where the judge actually hits target_accuracy.

    Returned from the labelled data rather than picked by hand, which is the
    whole point: the review threshold is a measurement, not a guess.
    Returns None when no cut reaches the target with enough support, which
    means the judge is not good enough to auto-approve anything yet.
    """
    if not pairs:
        return None
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0, 1]")

    for candidate in sorted({pair.judge_confidence for pair in pairs}):
        above = [pair for pair in pairs if pair.judge_confidence >= candidate]
        if len(above) >= min_support and agreement_rate(above) >= target_accuracy:
            return candidate
    return None


def _warnings(
    pairs: Sequence[LabelPair],
    bins: tuple[ReliabilityBin, ...],
    agreement: float,
    distinct: int,
) -> tuple[str, ...]:
    """Reasons not to believe this report.

    A calibration report that looks perfect is usually describing its own
    inputs rather than the judge. These checks exist so the report says so
    instead of being quoted as a finding.
    """
    notes: list[str] = []
    count = len(pairs)

    if count < MIN_RELIABLE_PAIRS:
        notes.append(
            f"only {count} labelled pairs, fewer than the {MIN_RELIABLE_PAIRS} "
            "these statistics need to mean much"
        )
    if distinct < count:
        notes.append(
            f"{count} pairs cover only {distinct} distinct cases, so repeated "
            "rows are being counted as independent observations"
        )
    if len(bins) < MIN_OCCUPIED_BINS:
        notes.append(
            "every judge confidence lands in one bin, so calibration cannot be "
            "measured across the range"
        )
    if count and agreement >= 1.0:
        notes.append(
            "perfect agreement, which usually means the labelled set contains "
            "nothing worth disagreeing about"
        )
    return tuple(notes)


def compute_calibration(
    pairs: Sequence[LabelPair],
    *,
    bins: int = DEFAULT_BINS,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> CalibrationReport:
    if not pairs:
        return CalibrationReport(warnings=("no labelled pairs",))

    threshold = recommend_threshold(
        pairs, target_accuracy=target_accuracy, min_support=min_support
    )
    above = (
        [pair for pair in pairs if pair.judge_confidence >= threshold]
        if threshold is not None
        else []
    )

    by_verdict: dict[str, float] = {}
    for verdict in JudgeVerdict:
        subset = [pair for pair in pairs if pair.judge_verdict == verdict]
        if subset:
            by_verdict[str(verdict)] = agreement_rate(subset)

    curve = reliability_bins(pairs, bins=bins)
    agreement = agreement_rate(pairs)
    distinct = len({pair.case_id for pair in pairs})

    return CalibrationReport(
        pairs=len(pairs),
        distinct_cases=distinct,
        agreement=agreement,
        kappa=cohens_kappa(pairs),
        brier=brier_score(pairs),
        expected_calibration_error=expected_calibration_error(pairs, bins=bins),
        mean_confidence=sum(p.judge_confidence for p in pairs) / len(pairs),
        bins=curve,
        agreement_by_judge_verdict=by_verdict,
        recommended_threshold=threshold,
        coverage_at_threshold=len(above) / len(pairs) if above else 0.0,
        accuracy_at_threshold=agreement_rate(above) if above else 0.0,
        warnings=_warnings(pairs, curve, agreement, distinct),
    )