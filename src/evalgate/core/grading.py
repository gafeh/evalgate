from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evalgate.core.dataset import GradingCase
from evalgate.core.scorers import NumericTolerance, SetF1
from evalgate.core.types import Score

SET_F1_WEIGHT = 0.6
POINT_TOTAL_WEIGHT = 0.4


class ModelGrading(BaseModel):
    """What the system under evaluation is contracted to return."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    awarded_item_ids: frozenset[str]
    total_points: float = Field(ge=0)
    feedback: str = ""

    @staticmethod
    def output_schema() -> dict[str, Any]:
        """JSON Schema for the output contract, used by JsonSchemaValid."""
        return ModelGrading.model_json_schema()


@dataclass(frozen=True, slots=True)
class GradingEvaluation:
    case_id: str
    scores: tuple[Score, ...]
    overall: float
    over_credit_points: float
    under_credit_points: float
    hallucinated_item_ids: tuple[str, ...]

    @property
    def exact_rubric_match(self) -> bool:
        return self.over_credit_points == 0.0 and self.under_credit_points == 0.0


def evaluate_grading(
    case: GradingCase,
    grading: ModelGrading,
    *,
    point_tolerance: float = 0.0,
) -> GradingEvaluation:
    """Score one model grading against ground truth.

    Over-credit and under-credit are reported separately rather than netted,
    because they are not equally bad. Awarding a point the student did not
    earn passes a wrong answer through to a real transcript, while missing a
    point gets caught the moment the student complains.
    """
    expected_ids = case.expected.awarded_item_ids
    actual_ids = grading.awarded_item_ids
    known_ids = case.rubric.item_ids

    f1 = SetF1().score(expected_ids, actual_ids)
    points = NumericTolerance(abs_tol=point_tolerance).score(
        case.expected_points, grading.total_points
    )

    over = case.rubric.points_for(frozenset(actual_ids - expected_ids))
    under = case.rubric.points_for(frozenset(expected_ids - actual_ids))
    hallucinated = tuple(sorted(actual_ids - known_ids))

    overall = SET_F1_WEIGHT * f1.value + POINT_TOTAL_WEIGHT * points.value

    return GradingEvaluation(
        case_id=case.id,
        scores=(f1, points),
        overall=overall,
        over_credit_points=over,
        under_credit_points=under,
        hallucinated_item_ids=hallucinated,
    )