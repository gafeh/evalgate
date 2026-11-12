from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Uuid as SAUuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalgate.core.judgment import JudgeVerdict
from evalgate.db.base import Base


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkItemStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"
    DEAD = "dead"


class CallPurpose(StrEnum):
    GRADER = "grader"
    JUDGE = "judge"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    LABELLED = "labelled"
    SKIPPED = "skipped"


class ReviewReason(StrEnum):
    """Why a result is in the queue.

    CALIBRATION rows are a random sample and are the only ones the
    calibration metrics are computed from. The rest are triage: they are
    there because something looks wrong, which makes them a biased sample.
    """

    CALIBRATION = "calibration"
    LOW_CONFIDENCE = "low_confidence"
    SAFETY = "safety"
    UNPARSEABLE = "unparseable"


def _enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """VARCHAR plus CHECK rather than a native Postgres enum type."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        length=16,
        values_callable=lambda cls: [member.value for member in cls],
        validate_strings=True,
    )


class UuidPk:
    id: Mapped[uuid.UUID] = mapped_column(SAUuid, primary_key=True, default=uuid.uuid4)


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)


class Dataset(UuidPk, Timestamped, Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("name", "version"),)

    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64))
    domain: Mapped[str] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    case_count: Mapped[int] = mapped_column(Integer)

    cases: Mapped[list[Case]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="raise",
    )


class Case(UuidPk, Timestamped, Base):
    __tablename__ = "cases"
    __table_args__ = (UniqueConstraint("dataset_id", "external_id"),)

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    external_id: Mapped[str] = mapped_column(String(128))
    question_id: Mapped[str] = mapped_column(String(128), index=True)
    question_prompt: Mapped[str] = mapped_column(Text)
    student_response: Mapped[str] = mapped_column(Text)
    rubric: Mapped[dict[str, Any]]
    expected: Mapped[dict[str, Any]]
    tags: Mapped[dict[str, Any]]

    dataset: Mapped[Dataset] = relationship(back_populates="cases", lazy="raise")


class Run(UuidPk, Timestamped, Base):
    __tablename__ = "runs"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT"), index=True
    )
    # Pinned at creation. If the dataset is later edited, this run still
    # records which bytes it actually evaluated.
    dataset_content_hash: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    params: Mapped[dict[str, Any]]
    label: Mapped[str | None] = mapped_column(String(128), default=None)
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status"), default=RunStatus.PENDING, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(default=None)
    finished_at: Mapped[datetime | None] = mapped_column(default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)


class WorkItem(UuidPk, Timestamped, Base):
    """One case waiting to be evaluated in one run. This is the queue."""

    __tablename__ = "work_items"
    __table_args__ = (
        UniqueConstraint("run_id", "case_id"),
        Index("ix_work_items_claimable", "run_id", "status", "available_at"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"))
    status: Mapped[WorkItemStatus] = mapped_column(
        _enum(WorkItemStatus, "work_item_status"), default=WorkItemStatus.QUEUED
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    claimed_by: Mapped[str | None] = mapped_column(String(128), default=None)
    claimed_at: Mapped[datetime | None] = mapped_column(default=None)
    heartbeat_at: Mapped[datetime | None] = mapped_column(default=None)
    # Retry backoff. A failed item is not claimable again until this passes.
    available_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, default=None)


class Result(UuidPk, Timestamped, Base):
    __tablename__ = "results"
    __table_args__ = (UniqueConstraint("run_id", "case_id"),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"))

    raw_output: Mapped[str] = mapped_column(Text)
    parsed_ok: Mapped[bool] = mapped_column(default=False)
    parse_error: Mapped[str | None] = mapped_column(Text, default=None)
    grading: Mapped[dict[str, Any] | None] = mapped_column(default=None)

    scores: Mapped[dict[str, Any]]
    overall: Mapped[float | None] = mapped_column(Float, default=None)
    over_credit_points: Mapped[float | None] = mapped_column(Float, default=None)
    under_credit_points: Mapped[float | None] = mapped_column(Float, default=None)
    hallucinated_item_ids: Mapped[dict[str, Any]]

    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    # Fingerprint of the scored outcome. Phase 9 chains these into an audit log.
    content_hash: Mapped[str] = mapped_column(String(64))


class LlmInvocation(UuidPk, Timestamped, Base):
    """Content-addressed response cache, shared across every run.

    Keyed by everything that determines the answer, so an identical prompt
    to an identical model is never paid for twice.
    """

    __tablename__ = "llm_invocations"
    __table_args__ = (UniqueConstraint("cache_key"),)

    cache_key: Mapped[str] = mapped_column(String(64))
    purpose: Mapped[CallPurpose] = mapped_column(_enum(CallPurpose, "call_purpose"))
    model: Mapped[str] = mapped_column(String(128))
    request: Mapped[dict[str, Any]]
    response_payload: Mapped[dict[str, Any] | None] = mapped_column(default=None)
    raw_text: Mapped[str] = mapped_column(Text)
    parse_error: Mapped[str | None] = mapped_column(Text, default=None)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    stop_reason: Mapped[str | None] = mapped_column(String(64), default=None)


class LlmCall(UuidPk, Timestamped, Base):
    """Append-only ledger of every call a run made, cache hits included.

    Separate from the cache so a rerun can report truthfully: the same work
    happened, and it cost nothing the second time.
    """

    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_run_purpose", "run_id", "purpose"),)

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    purpose: Mapped[CallPurpose] = mapped_column(_enum(CallPurpose, "call_purpose"))
    model: Mapped[str] = mapped_column(String(128))
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    cached: Mapped[bool] = mapped_column(default=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)


class JudgeAssessmentRow(UuidPk, Timestamped, Base):
    __tablename__ = "judge_assessments"
    __table_args__ = (UniqueConstraint("result_id"),)

    result_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("results.id", ondelete="CASCADE")
    )
    model: Mapped[str] = mapped_column(String(128))
    verdict: Mapped[JudgeVerdict] = mapped_column(_enum(JudgeVerdict, "judge_verdict"))
    confidence: Mapped[float] = mapped_column(Float)
    reasoning: Mapped[str] = mapped_column(Text, default="")
    criteria: Mapped[dict[str, Any]]
    cache_key: Mapped[str] = mapped_column(String(64))


class ReviewItem(UuidPk, Timestamped, Base):
    """One result queued for a human to look at.

    A result can be queued once. The calibration sample is drawn before any
    triage, so it stays a random sample of all judged results rather than
    whatever triage left over.
    """

    __tablename__ = "review_items"
    __table_args__ = (
        UniqueConstraint("result_id"),
        Index("ix_review_items_pending", "status", "priority"),
    )

    result_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("results.id", ondelete="CASCADE")
    )
    status: Mapped[ReviewStatus] = mapped_column(
        _enum(ReviewStatus, "review_status"), default=ReviewStatus.PENDING
    )
    reason: Mapped[ReviewReason] = mapped_column(_enum(ReviewReason, "review_reason"))
    priority: Mapped[float] = mapped_column(Float, default=0.0)


class HumanLabel(UuidPk, Timestamped, Base):
    """A human's verdict on the grader's feedback, on the judge's own scale.

    Same three values as the judge uses, deliberately. You can only measure
    agreement between two raters answering the same question.
    """

    __tablename__ = "human_labels"
    __table_args__ = (UniqueConstraint("review_item_id"),)

    review_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_items.id", ondelete="CASCADE")
    )
    labeller: Mapped[str] = mapped_column(String(64), index=True)
    verdict: Mapped[JudgeVerdict] = mapped_column(_enum(JudgeVerdict, "judge_verdict"))
    criteria: Mapped[dict[str, Any]]
    notes: Mapped[str] = mapped_column(Text, default="")
    seconds_spent: Mapped[int] = mapped_column(Integer, default=0)
class AuditEntry(UuidPk, Timestamped, Base):
    """One link in a run's hash chain, written when the run is sealed.

    Written in a single pass rather than appended per result: keeping seq
    contiguous while several workers commit concurrently would need a lock on
    every write.
    """

    __tablename__ = "audit_entries"
    __table_args__ = (UniqueConstraint("run_id", "seq"),)

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    seq: Mapped[int] = mapped_column(Integer)
    entry_type: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]]
    prev_hash: Mapped[str] = mapped_column(String(64))
    entry_hash: Mapped[str] = mapped_column(String(64), index=True)