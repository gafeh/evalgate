from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evalgate.core.dataset import GradingCase
from evalgate.core.grading import ModelGrading


@dataclass(frozen=True, slots=True)
class GradeOutput:
    """What came back from the system under evaluation.

    raw_output is always kept, even when parsing failed, because the text a
    model returned instead of valid JSON is the most useful debugging artefact
    there is.
    """

    raw_output: str
    grading: ModelGrading | None = None
    parse_error: str | None = None
    latency_ms: int = 0


class Grader(Protocol):
    @property
    def model_name(self) -> str: ...

    async def grade(self, case: GradingCase) -> GradeOutput: ...