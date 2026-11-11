from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from evalgate.core.scorers.base import Scorer, register
from evalgate.core.types import Score


@register
@dataclass(frozen=True, slots=True)
class NumericTolerance(Scorer[float, float]):
    """Equality within tolerance, with the signed error kept in the detail.

    Signed error matters for grading: a model that consistently awards
    too many points is a different failure from one that is merely noisy,
    and you can only see that if the sign survives aggregation.
    """

    name: ClassVar[str] = "numeric_tolerance"
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    def score(self, expected: float, actual: float) -> Score:
        hit = math.isclose(actual, expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        signed_error = actual - expected
        return Score(
            scorer=self.name,
            value=1.0 if hit else 0.0,
            passed=hit,
            detail={
                "expected": expected,
                "actual": actual,
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
            },
        )