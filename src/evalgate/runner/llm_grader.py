from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.core.dataset import GradingCase
from evalgate.core.grading import ModelGrading
from evalgate.db.models import CallPurpose
from evalgate.db.session import session_scope
from evalgate.judge.cache import complete_with_cache
from evalgate.judge.client import LlmClient, LlmRequest
from evalgate.judge.prompts import (
    GRADER_SCHEMA,
    GRADER_TOOL_DESCRIPTION,
    GRADER_TOOL_NAME,
    grader_messages,
)
from evalgate.runner.protocol import GradeOutput


@dataclass(frozen=True, slots=True)
class LlmGrader:
    """The system under evaluation.

    Holds its own engine rather than sharing the worker's session, so cache
    and ledger writes commit independently of the work queue's state.
    """

    client: LlmClient
    engine: AsyncEngine
    run_id: UUID
    model: str
    max_tokens: int = 1024

    @property
    def model_name(self) -> str:
        return self.model

    async def grade(self, case: GradingCase) -> GradeOutput:
        system, user = grader_messages(case)
        request = LlmRequest(
            model=self.model,
            system=system,
            user=user,
            tool_name=GRADER_TOOL_NAME,
            tool_description=GRADER_TOOL_DESCRIPTION,
            input_schema=GRADER_SCHEMA,
            max_tokens=self.max_tokens,
        )

        async with session_scope(self.engine) as session:
            response, _cached = await complete_with_cache(
                session,
                self.client,
                request,
                purpose=CallPurpose.GRADER,
                run_id=self.run_id,
            )

        if response.payload is None:
            return GradeOutput(
                raw_output=response.raw_text,
                parse_error=response.parse_error or "no payload",
                latency_ms=response.latency_ms,
            )

        try:
            grading = ModelGrading(
                awarded_item_ids=frozenset(response.payload["awarded_item_ids"]),
                total_points=float(response.payload["total_points"]),
                feedback=str(response.payload.get("feedback", "")),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            return GradeOutput(
                raw_output=response.raw_text,
                parse_error=f"payload did not fit the contract: {exc}",
                latency_ms=response.latency_ms,
            )

        return GradeOutput(
            raw_output=response.raw_text,
            grading=grading,
            latency_ms=response.latency_ms,
        )