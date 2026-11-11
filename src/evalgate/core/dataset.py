from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalgate.core.hashing import content_hash

_STRICT = ConfigDict(frozen=True, extra="forbid")


class RubricItem(BaseModel):
    """One scoring point from an AP CS free-response rubric."""

    model_config = _STRICT

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    points: float = Field(default=1.0, gt=0)


class Rubric(BaseModel):
    model_config = _STRICT

    items: tuple[RubricItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Rubric:
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric item ids must be unique")
        return self

    @property
    def item_ids(self) -> frozenset[str]:
        return frozenset(item.id for item in self.items)

    @property
    def max_points(self) -> float:
        return sum(item.points for item in self.items)

    def points_for(self, item_ids: frozenset[str]) -> float:
        return sum(item.points for item in self.items if item.id in item_ids)


class ExpectedGrading(BaseModel):
    """Ground truth for one case, authored by a human.

    must_mention and must_not_mention are the feedback requirements the
    LLM judge checks in Phase 4. They stay out of the deterministic
    scorers on purpose.
    """

    model_config = _STRICT

    awarded_item_ids: frozenset[str]
    notes: str = ""
    must_mention: tuple[str, ...] = ()
    must_not_mention: tuple[str, ...] = ()


class GradingCase(BaseModel):
    model_config = _STRICT

    id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question_prompt: str = Field(min_length=1)
    rubric: Rubric
    student_response: str
    expected: ExpectedGrading
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _expected_within_rubric(self) -> GradingCase:
        unknown = self.expected.awarded_item_ids - self.rubric.item_ids
        if unknown:
            raise ValueError(
                f"case {self.id}: expected awards unknown rubric items {sorted(unknown)}"
            )
        return self

    @property
    def expected_points(self) -> float:
        return self.rubric.points_for(self.expected.awarded_item_ids)


class Dataset(BaseModel):
    model_config = _STRICT

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    domain: str = "apcs_frq"
    cases: tuple[GradingCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_case_ids(self) -> Dataset:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique within a dataset")
        return self

    def content_hash(self) -> str:
        """Stable fingerprint. Two datasets with the same hash are the same dataset."""
        return content_hash(self.model_dump())

    def slice_values(self, tag: str) -> frozenset[str]:
        return frozenset(case.tags[tag] for case in self.cases if tag in case.tags)