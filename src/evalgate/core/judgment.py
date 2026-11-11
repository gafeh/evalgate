from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JudgeVerdict(StrEnum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


CRITERIA = (
    "accurate",
    "actionable",
    "covers_required_points",
    "no_full_solution",
    "consistent_with_award",
)


class JudgeAssessment(BaseModel):
    """A judge's opinion of one piece of grader feedback.

    confidence is the judge's own claim about itself. It is stored as given
    and never trusted until it has been checked against human labels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reasoning: str = ""
    verdict: JudgeVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    criteria: dict[str, bool] = Field(default_factory=dict)

    @field_validator("criteria")
    @classmethod
    def _only_known_criteria(cls, value: dict[str, bool]) -> dict[str, bool]:
        return {key: bool(value.get(key, False)) for key in CRITERIA}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> JudgeAssessment:
        """Build from a raw tool payload, tolerating missing optional fields."""
        return cls(
            reasoning=str(payload.get("reasoning", "")),
            verdict=JudgeVerdict(payload["verdict"]),
            confidence=float(payload["confidence"]),
            criteria=dict(payload.get("criteria") or {}),
        )
