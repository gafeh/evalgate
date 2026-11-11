from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.core.dataset import GradingCase
from evalgate.core.grading import ModelGrading
from evalgate.core.judgment import JudgeAssessment
from evalgate.db.models import CallPurpose
from evalgate.db.session import session_scope
from evalgate.judge.cache import complete_with_cache
from evalgate.judge.client import LlmClient, LlmRequest
from evalgate.judge.prompts import (
    JUDGE_SCHEMA,
    JUDGE_TOOL_DESCRIPTION,
    JUDGE_TOOL_NAME,
    judge_messages,
)


@dataclass(frozen=True, slots=True)
class AssessmentResult:
    assessment: JudgeAssessment
    cache_key: str
    model: str


@dataclass(frozen=True, slots=True)
class LlmJudge:
    client: LlmClient
    engine: AsyncEngine
    run_id: UUID
    model: str
    max_tokens: int = 1024
    # Some models reject tool_choice type "tool". Turn this off for those and
    # accept that a judgment can come back missing.
    force_tool: bool = True

    @property
    def model_name(self) -> str:
        return self.model

    async def assess(
        self, case: GradingCase, grading: ModelGrading
    ) -> AssessmentResult | None:
        """Judge one piece of feedback. Returns None when the judge misbehaved."""
        system, user = judge_messages(case, grading)
        request = LlmRequest(
            model=self.model,
            system=system,
            user=user,
            tool_name=JUDGE_TOOL_NAME,
            tool_description=JUDGE_TOOL_DESCRIPTION,
            input_schema=JUDGE_SCHEMA,
            max_tokens=self.max_tokens,
            force_tool=self.force_tool,
        )

        async with session_scope(self.engine) as session:
            response, _cached = await complete_with_cache(
                session,
                self.client,
                request,
                purpose=CallPurpose.JUDGE,
                run_id=self.run_id,
            )

        if response.payload is None:
            return None

        try:
            assessment = JudgeAssessment.from_payload(response.payload)
        except (KeyError, TypeError, ValueError, ValidationError):
            return None

        return AssessmentResult(
            assessment=assessment,
            cache_key=request.cache_key(),
            model=self.model,
        )