from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

OUT = ConfigDict(frozen=True)


class DatasetOut(BaseModel):
    model_config = OUT

    id: UUID
    name: str
    version: str
    domain: str
    content_hash: str
    case_count: int
    created_at: datetime


class CreateRunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_version: str
    model: str
    label: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RunOut(BaseModel):
    model_config = OUT

    id: UUID
    dataset_id: UUID
    dataset_content_hash: str
    model: str
    label: str | None
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class UsageOut(BaseModel):
    model_config = OUT

    calls: int
    cache_hits: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class SummaryOut(BaseModel):
    model_config = OUT

    cases: int
    unparseable: int
    mean_overall: float
    exact_match_rate: float
    over_credit_points: float
    under_credit_points: float


class SliceOut(BaseModel):
    model_config = OUT

    tag_value: str
    cases: int
    mean_overall: float
    over_credit_points: float


class RunDetailOut(BaseModel):
    model_config = OUT

    run: RunOut
    work_items: dict[str, int]
    summary: SummaryOut
    usage: UsageOut
    slices: list[SliceOut]


class ResultOut(BaseModel):
    model_config = OUT

    case_id: str
    parsed_ok: bool
    overall: float | None
    over_credit_points: float | None
    under_credit_points: float | None
    judge_verdict: str | None
    judge_confidence: float | None
    feedback: str


class IntervalOut(BaseModel):
    model_config = OUT

    point: float
    low: float
    high: float
    confidence: float


class CompareOut(BaseModel):
    model_config = OUT

    verdict: str
    reason: str
    passed: bool
    paired_cases: int
    baseline: SummaryOut
    candidate: SummaryOut
    overall_delta: IntervalOut | None
    over_credit_delta: IntervalOut | None
    cases_needed: int | None


class EnqueueIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample: int = 50
    seed: int = 0
    triage: bool = True


class EnqueueOut(BaseModel):
    model_config = OUT

    added_to_sample: int
    triaged: int
    queue: dict[str, int]


class ReviewTaskOut(BaseModel):
    model_config = OUT

    review_item_id: UUID
    reason: str
    case_id: str
    question_prompt: str
    rubric_lines: list[str]
    student_response: str
    expected_awards: list[str]
    expected_points: float
    grader_awards: list[str]
    grader_points: float
    grader_feedback: str
    must_mention: list[str]
    must_not_mention: list[str]
    parsed_ok: bool


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labeller: str
    verdict: str
    criteria: dict[str, bool] = Field(default_factory=dict)
    notes: str = ""
    seconds_spent: int = 0


class BinOut(BaseModel):
    model_config = OUT

    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float
    gap: float


class CalibrationOut(BaseModel):
    model_config = OUT

    pairs: int
    distinct_cases: int
    agreement: float
    kappa: float
    brier: float
    expected_calibration_error: float
    mean_confidence: float
    overconfident: bool
    trustworthy: bool
    warnings: list[str]
    bins: list[BinOut]
    agreement_by_judge_verdict: dict[str, float]
    recommended_threshold: float | None
    coverage_at_threshold: float
    accuracy_at_threshold: float