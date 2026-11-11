from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Score:
    """The result of one scorer on one case.

    value is always normalised to [0, 1] so unrelated scorers can be
    aggregated. passed is the scorer's own verdict, which is not always
    value == 1.0 (a set-overlap scorer may pass at 0.8).
    """

    scorer: str
    value: float
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scorer:
            raise ValueError("scorer name must not be empty")
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"score value must be in [0, 1], got {self.value}")