from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evalgate.core.outcome import CaseOutcome
from evalgate.core.stats import mean

EMPTY_TAG = "(untagged)"


@dataclass(frozen=True, slots=True)
class RunSummary:
    cases: int = 0
    unparseable: int = 0
    mean_overall: float = 0.0
    exact_match_rate: float = 0.0
    over_credit_points: float = 0.0
    under_credit_points: float = 0.0
    mean_over_credit: float = 0.0
    mean_under_credit: float = 0.0

    @property
    def unparseable_rate(self) -> float:
        return self.unparseable / self.cases if self.cases else 0.0


def summarize(outcomes: Sequence[CaseOutcome]) -> RunSummary:
    if not outcomes:
        return RunSummary()

    count = len(outcomes)
    over = sum(outcome.over_credit_points for outcome in outcomes)
    under = sum(outcome.under_credit_points for outcome in outcomes)

    return RunSummary(
        cases=count,
        unparseable=sum(1 for outcome in outcomes if not outcome.parsed_ok),
        mean_overall=mean([outcome.effective_overall for outcome in outcomes]),
        exact_match_rate=sum(1 for o in outcomes if o.exact_match) / count,
        over_credit_points=over,
        under_credit_points=under,
        mean_over_credit=over / count,
        mean_under_credit=under / count,
    )


def summarize_by_slice(
    outcomes: Sequence[CaseOutcome], tag: str
) -> dict[str, RunSummary]:
    """Group by a tag value. Cases missing the tag land under EMPTY_TAG.

    Slices are where a regression actually lives. An aggregate that moved by
    half a point often means one error mode fell off a cliff while the rest
    held steady.
    """
    buckets: dict[str, list[CaseOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.tags.get(tag, EMPTY_TAG), []).append(outcome)
    return {key: summarize(value) for key, value in sorted(buckets.items())}