from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from evalgate.core.dataset import GradingCase
from evalgate.core.grading import ModelGrading
from evalgate.runner.protocol import GradeOutput


@dataclass(frozen=True, slots=True)
class StubGrader:
    """A fake grader with dialable failure modes.

    Behaviour is seeded per case id, so the same case always gets the same
    treatment. That determinism is what lets you write assertions about a
    run's aggregate scores.
    """

    model_name: str = "stub-v1"
    over_credit_rate: float = 0.0
    under_credit_rate: float = 0.0
    malformed_rate: float = 0.0
    raise_rate: float = 0.0
    latency_ms: int = 0

    def _rng(self, case: GradingCase) -> random.Random:
        return random.Random(f"{self.model_name}:{case.id}")

    async def grade(self, case: GradingCase) -> GradeOutput:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)

        roll = self._rng(case).random()
        if roll < self.raise_rate:
            raise RuntimeError(f"stub grader exploded on {case.id}")

        if roll < self.raise_rate + self.malformed_rate:
            return GradeOutput(
                raw_output="Sure! Happy to grade this response for you.",
                parse_error="no json object in response",
                latency_ms=self.latency_ms,
            )

        awarded = case.expected.awarded_item_ids
        if roll < self.raise_rate + self.malformed_rate + self.over_credit_rate:
            awarded = case.rubric.item_ids
        elif roll < (
            self.raise_rate
            + self.malformed_rate
            + self.over_credit_rate
            + self.under_credit_rate
        ):
            awarded = frozenset(sorted(awarded)[:-1]) if awarded else awarded

        grading = ModelGrading(
            awarded_item_ids=awarded,
            total_points=case.rubric.points_for(awarded),
            feedback=f"Stub feedback for {case.id}.",
        )
        return GradeOutput(
            raw_output=grading.model_dump_json(),
            grading=grading,
            latency_ms=self.latency_ms,
        )