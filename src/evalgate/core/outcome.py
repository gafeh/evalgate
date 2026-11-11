from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """One scored case, flattened for analysis. Pure data, no database types."""

    case_id: str
    parsed_ok: bool
    overall: float | None
    over_credit_points: float = 0.0
    under_credit_points: float = 0.0
    exact_match: bool = False
    tags: Mapping[str, str] = field(default_factory=dict)

    @property
    def effective_overall(self) -> float:
        """Unparseable output scores zero rather than being skipped.

        Excluding it would let a model that fails to produce valid output
        half the time report the average of only its successes.
        """
        return self.overall if self.overall is not None else 0.0