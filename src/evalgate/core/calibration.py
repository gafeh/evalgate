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
    """One case where both the judge and a human gave a verdict.

    weight is the inverse of this case's inclusion probability. Under a simple
    random sample every weight is 1 and the weighted statistics collapse to the
    ordinary ones. Under stratified sampling a rare stratum is deliberately
    oversampled, and the weight is what stops that oversampling from distorting
    the estimate.
    """

    case_id: str
    judge_verdict: JudgeVerdict
    judge_confidence: float
    human_verdict: JudgeVerdict
    weight: float = 1.0
    stratum: str = ""

    @property
    def agrees(self) -> bool:
        return self.judge_verdict == self.human_verdict


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    low: float
    high: float
    count: int
    weight: float
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Positive means underconfident, negative means overconfident."""
        return self.accuracy - self.mean_confidence


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    pairs: int = 0
    effective_pairs: float = 0.0
    distinct_cases: int = 0
    weighted: bool = False
    strata: dict[str, int] = field(default_factory=dict)
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
    pairs_below_threshold: int = 0
    accuracy_below_threshold: float = 0.0
    warnings: tuple[str, ...] = ()
    threshold_warnings: tuple[str, ...] = ()

    @property
    def overconfident(self) -> bool:
        return self.mean_confidence > self.agreement

    @property
    def trustworthy(self) -> bool:
        """Whether the agreement statistics should be quoted as a measurement.

        Scoped to the statistics. The threshold recommendation has its own
        warnings, because it can be unsupported even when these are sound.
        """
        return not self.warnings


def total_weight(pairs: Sequence[LabelPair]) -> float:
    return sum(pair.weight for pair in pairs)


def effective_sample_size(pairs: Sequence[LabelPair]) -> float:
    """Kish's effective sample size.

    Unequal weights buy coverage of rare strata at the cost of precision. This
    says what the weighted sample is worth in equally-weighted observations, so
    "50 labels" does not get quoted when the weights make it worth 31.
    """
    if not pairs:
        return 0.0
    squares = sum(pair.weight**2 for pair in pairs)
    return (total_weight(pairs) ** 2) / squares if squares else 0.0


def agreement_rate(pairs: Sequence[LabelPair]) -> float:
    weight = total_weight(pairs)
    if weight <= 0.0:
        return 0.0
    return sum(pair.weight for pair in pairs if pair.agrees) / weight


def cohens_kappa(pairs: Sequence[LabelPair]) -> float:
    """Agreement corrected for what chance alone would produce.

    Raw agreement flatters a judge that always says "pass" on a mostly-passing
    set. Kappa removes that: zero means no better than guessing with the same
    marginal frequencies.
    """
    weight = total_weight(pairs)
    if weight <= 0.0:
        return 0.0

    observed = agreement_rate(pairs)
    expected = 0.0
    for verdict in JudgeVerdict:
        judge_share = (
            sum(p.weight for p in pairs if p.judge_verdict == verdict) / weight
        )
        human_share = (
            sum(p.weight for p in pairs if p.human_verdict == verdict) / weight
        )
        expected += judge_share * human_share

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def brier_score(pairs: Sequence[LabelPair]) -> float:
    """Mean squared error of confidence against being right. Lower is better.

    This is the number that tests whether confidence is a probability or just
    a vibe. A judge claiming 0.9 should be wrong about one time in ten.
    """
    weight = total_weight(pairs)
    if weight <= 0.0:
        return 0.0
    return (
        sum(
            pair.weight * (pair.judge_confidence - (1.0 if pair.agrees else 0.0)) ** 2
            for pair in pairs
        )
        / weight
    )


def mean_confidence(pairs: Sequence[LabelPair]) -> float:
    weight = total_weight(pairs)
    if weight <= 0.0:
        return 0.0
    return sum(pair.weight * pair.judge_confidence for pair in pairs) / weight


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
                weight=total_weight(bucket),
                mean_confidence=mean_confidence(bucket),
                accuracy=agreement_rate(bucket),
            )
        )
    return tuple(out)


def expected_calibration_error(
    pairs: Sequence[LabelPair], *, bins: int = DEFAULT_BINS
) -> float:
    """Weighted average gap between claimed confidence and actual accuracy."""
    weight = total_weight(pairs)
    if weight <= 0.0:
        return 0.0
    return sum(
        (item.weight / weight) * abs(item.accuracy - item.mean_confidence)
        for item in reliability_bins(pairs, bins=bins)
    )


def recommend_threshold(
    pairs: Sequence[LabelPair],
    *,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> float | None:
    """Lowest confidence cut where the judge actually hits target_accuracy.

    Support is counted in effective observations rather than rows, so a cut
    held up by a handful of heavily-weighted cases does not qualify.
    """
    if not pairs:
        return None
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0, 1]")

    for candidate in sorted({pair.judge_confidence for pair in pairs}):
        above = [pair for pair in pairs if pair.judge_confidence >= candidate]
        if (
            effective_sample_size(above) >= min_support
            and agreement_rate(above) >= target_accuracy
        ):
            return candidate
    return None


def _warnings(
    pairs: Sequence[LabelPair],
    bins: tuple[ReliabilityBin, ...],
    agreement: float,
    distinct: int,
    effective: float,
) -> tuple[str, ...]:
    """Reasons not to believe the agreement statistics."""
    notes: list[str] = []
    count = len(pairs)

    if effective < MIN_RELIABLE_PAIRS:
        notes.append(
            f"effective sample size {effective:.1f}, fewer than the "
            f"{MIN_RELIABLE_PAIRS} these statistics need to mean much"
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


def _threshold_warnings(
    threshold: float | None,
    below: Sequence[LabelPair],
    coverage: float,
    min_support: int,
) -> tuple[str, ...]:
    """Reasons not to act on the recommended threshold.

    A cut only means something if the data has cases on both sides of it.
    """
    if threshold is None:
        return ()

    notes: list[str] = []
    if coverage >= 1.0:
        notes.append(
            "this cut sends nothing to review, so it is not a working "
            "threshold. On this sample the judge clears the target everywhere, "
            "which is also what too few low-confidence labels looks like"
        )
    if effective_sample_size(below) < min_support:
        notes.append(
            f"only {len(below)} labelled pairs fall below the cut, so its low "
            "side is unmeasured. Sample the low-confidence strata before "
            "trusting it"
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
    below = (
        [pair for pair in pairs if pair.judge_confidence < threshold]
        if threshold is not None
        else []
    )

    by_verdict: dict[str, float] = {}
    for verdict in JudgeVerdict:
        subset = [pair for pair in pairs if pair.judge_verdict == verdict]
        if subset:
            by_verdict[str(verdict)] = agreement_rate(subset)

    strata: dict[str, int] = {}
    for pair in pairs:
        if pair.stratum:
            strata[pair.stratum] = strata.get(pair.stratum, 0) + 1

    curve = reliability_bins(pairs, bins=bins)
    agreement = agreement_rate(pairs)
    distinct = len({pair.case_id for pair in pairs})
    effective = effective_sample_size(pairs)
    weight = total_weight(pairs)
    coverage = (total_weight(above) / weight) if above and weight else 0.0

    return CalibrationReport(
        pairs=len(pairs),
        effective_pairs=effective,
        distinct_cases=distinct,
        weighted=any(pair.weight != 1.0 for pair in pairs),
        strata=dict(sorted(strata.items())),
        agreement=agreement,
        kappa=cohens_kappa(pairs),
        brier=brier_score(pairs),
        expected_calibration_error=expected_calibration_error(pairs, bins=bins),
        mean_confidence=mean_confidence(pairs),
        bins=curve,
        agreement_by_judge_verdict=by_verdict,
        recommended_threshold=threshold,
        coverage_at_threshold=coverage,
        accuracy_at_threshold=agreement_rate(above) if above else 0.0,
        pairs_below_threshold=len(below),
        accuracy_below_threshold=agreement_rate(below) if below else 0.0,
        warnings=_warnings(pairs, curve, agreement, distinct, effective),
        threshold_warnings=_threshold_warnings(
            threshold, below, coverage, min_support
        ),
    )