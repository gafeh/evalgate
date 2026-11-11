from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from evalgate.core.aggregate import RunSummary, summarize, summarize_by_slice
from evalgate.core.outcome import CaseOutcome
from evalgate.core.stats import (
    DEFAULT_ITERATIONS,
    ConfidenceInterval,
    paired_delta_ci,
    standard_deviation,
    suggested_paired_sample_size,
)


class RegressionVerdict(StrEnum):
    IMPROVED = "improved"
    NO_CHANGE = "no_change"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Thresholds for the gate.

    min_effect is the smallest change in mean score worth acting on. It
    exists so the gate cannot be tripped by a difference too small to matter,
    however confidently it is measured.

    over_credit_min_effect is in rubric points per case, and is set tighter
    than min_effect on purpose: awarding points a student did not earn is a
    worse failure than scoring slightly lower overall.
    """

    confidence: float = 0.95
    min_effect: float = 0.02
    over_credit_min_effect: float = 0.05
    iterations: int = DEFAULT_ITERATIONS
    seed: int = 0


@dataclass(frozen=True, slots=True)
class SliceDelta:
    tag_value: str
    cases: int
    baseline_mean: float
    candidate_mean: float

    @property
    def delta(self) -> float:
        return self.candidate_mean - self.baseline_mean


@dataclass(frozen=True, slots=True)
class Comparison:
    verdict: RegressionVerdict
    reason: str
    paired_cases: int
    baseline: RunSummary
    candidate: RunSummary
    overall_delta: ConfidenceInterval | None
    over_credit_delta: ConfidenceInterval | None
    slices: tuple[SliceDelta, ...] = ()
    cases_needed: int | None = None

    @property
    def passed(self) -> bool:
        """Only a measured non-change or improvement passes.

        Inconclusive does not pass. "We could not tell" is not the same as
        "nothing changed", and a gate that conflates them waves regressions
        through whenever the dataset is small.
        """
        return self.verdict in (RegressionVerdict.NO_CHANGE, RegressionVerdict.IMPROVED)


def _slice_deltas(
    baseline: Sequence[CaseOutcome], candidate: Sequence[CaseOutcome], tag: str
) -> tuple[SliceDelta, ...]:
    base_slices = summarize_by_slice(baseline, tag)
    cand_slices = summarize_by_slice(candidate, tag)
    deltas = []
    for key in sorted(set(base_slices) | set(cand_slices)):
        base = base_slices.get(key, RunSummary())
        cand = cand_slices.get(key, RunSummary())
        deltas.append(
            SliceDelta(
                tag_value=key,
                cases=max(base.cases, cand.cases),
                baseline_mean=base.mean_overall,
                candidate_mean=cand.mean_overall,
            )
        )
    return tuple(deltas)


def compare_runs(
    baseline: Sequence[CaseOutcome],
    candidate: Sequence[CaseOutcome],
    *,
    config: GateConfig | None = None,
    slice_tag: str = "error_mode",
) -> Comparison:
    """Compare two runs over the same cases and return a gate verdict.

    Only cases present in both runs are compared, because an unpaired case
    contributes nothing to a paired test and silently shifts the means.
    """
    cfg = config or GateConfig()

    base_by_id = {outcome.case_id: outcome for outcome in baseline}
    cand_by_id = {outcome.case_id: outcome for outcome in candidate}
    shared = sorted(set(base_by_id) & set(cand_by_id))

    if not shared:
        return Comparison(
            verdict=RegressionVerdict.INCONCLUSIVE,
            reason="the two runs share no cases",
            paired_cases=0,
            baseline=summarize(baseline),
            candidate=summarize(candidate),
            overall_delta=None,
            over_credit_delta=None,
        )

    paired_base = [base_by_id[case_id] for case_id in shared]
    paired_cand = [cand_by_id[case_id] for case_id in shared]

    overall = paired_delta_ci(
        [outcome.effective_overall for outcome in paired_base],
        [outcome.effective_overall for outcome in paired_cand],
        confidence=cfg.confidence,
        iterations=cfg.iterations,
        seed=cfg.seed,
    )
    over_credit = paired_delta_ci(
        [outcome.over_credit_points for outcome in paired_base],
        [outcome.over_credit_points for outcome in paired_cand],
        confidence=cfg.confidence,
        iterations=cfg.iterations,
        seed=cfg.seed,
    )

    deltas = [
        cand.effective_overall - base.effective_overall
        for base, cand in zip(paired_base, paired_cand, strict=True)
    ]

    verdict, reason, cases_needed = _decide(overall, over_credit, deltas, cfg)

    return Comparison(
        verdict=verdict,
        reason=reason,
        paired_cases=len(shared),
        baseline=summarize(paired_base),
        candidate=summarize(paired_cand),
        overall_delta=overall,
        over_credit_delta=over_credit,
        slices=_slice_deltas(paired_base, paired_cand, slice_tag),
        cases_needed=cases_needed,
    )


def _decide(
    overall: ConfidenceInterval,
    over_credit: ConfidenceInterval,
    deltas: Sequence[float],
    cfg: GateConfig,
) -> tuple[RegressionVerdict, str, int | None]:
    # Over-credit first. A candidate that scores better overall while handing
    # out more unearned points has still got worse at the thing that matters.
    if over_credit.low > cfg.over_credit_min_effect:
        return (
            RegressionVerdict.REGRESSED,
            (
                f"over-credit rose by {over_credit.point:+.3f} points per case "
                f"({over_credit.low:+.3f} at the low end of the interval)"
            ),
            None,
        )

    if overall.high < 0.0 and overall.point <= -cfg.min_effect:
        return (
            RegressionVerdict.REGRESSED,
            f"mean score fell by {overall} and the interval stays below zero",
            None,
        )

    if overall.low > 0.0 and overall.point >= cfg.min_effect:
        return (
            RegressionVerdict.IMPROVED,
            f"mean score rose by {overall} and the interval stays above zero",
            None,
        )

    # Equivalence, not a zero test. The question is whether the whole interval
    # fits inside the band we said we care about, which rules out a change of
    # min_effect in either direction. An interval of [-0.005, -0.005] is a
    # certain, trivial shift and belongs here, even though it excludes zero.
    if -cfg.min_effect <= overall.low and overall.high <= cfg.min_effect:
        return (
            RegressionVerdict.NO_CHANGE,
            (
                f"interval {overall} lies inside the minimum effect of "
                f"±{cfg.min_effect}"
            ),
            None,
        )

    needed = suggested_paired_sample_size(
        standard_deviation(deltas), cfg.min_effect, confidence=cfg.confidence
    )
    return (
        RegressionVerdict.INCONCLUSIVE,
        f"interval {overall} is too wide to rule out a change of {cfg.min_effect}",
        needed,
    )