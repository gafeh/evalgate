import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.core.grading import ModelGrading
from evalgate.core.judgment import CRITERIA, JudgeAssessment, JudgeVerdict
from evalgate.db.models import Run
from evalgate.db.repositories import create_run, save_dataset
from evalgate.db.session import session_scope
from evalgate.judge.assessor import LlmJudge
from tests.factories import make_case, make_dataset
from tests.fakes import FakeLlmClient

GOOD_PAYLOAD = {
    "reasoning": "The feedback names the loop bound and gives no code.",
    "verdict": "pass",
    "confidence": 0.82,
    "criteria": dict.fromkeys(CRITERIA, True),
}


async def _a_run(engine: AsyncEngine) -> Run:
    async with session_scope(engine) as session:
        dataset = await save_dataset(session, make_dataset(make_case("a")))
        return await create_run(session, dataset=dataset, model="claude-haiku-5-5")


def test_assessment_rejects_impossible_confidence() -> None:
    with pytest.raises(ValidationError):
        JudgeAssessment(verdict=JudgeVerdict.PASS, confidence=1.4)


def test_assessment_fills_missing_criteria_as_false() -> None:
    assessment = JudgeAssessment.from_payload(
        {"verdict": "partial", "confidence": 0.5, "criteria": {"accurate": True}}
    )
    assert assessment.criteria["accurate"] is True
    assert assessment.criteria["actionable"] is False
    assert set(assessment.criteria) == set(CRITERIA)


def test_assessment_ignores_criteria_it_does_not_know() -> None:
    assessment = JudgeAssessment.from_payload(
        {"verdict": "pass", "confidence": 0.9, "criteria": {"vibes": True}}
    )
    assert "vibes" not in assessment.criteria


async def test_judge_returns_a_parsed_assessment(engine: AsyncEngine) -> None:
    run = await _a_run(engine)
    judge = LlmJudge(
        client=FakeLlmClient(payload=GOOD_PAYLOAD),
        engine=engine,
        run_id=run.id,
        model="claude-sonnet-5-5",
    )
    grading = ModelGrading(
        awarded_item_ids=frozenset({"r1", "r3"}), total_points=2.0, feedback="Check the bound."
    )

    assessed = await judge.assess(make_case("a"), grading)

    assert assessed is not None
    assert assessed.assessment.verdict is JudgeVerdict.PASS
    assert assessed.assessment.confidence == 0.82
    assert len(assessed.cache_key) == 64


async def test_judge_returns_none_on_a_malformed_payload(engine: AsyncEngine) -> None:
    run = await _a_run(engine)
    judge = LlmJudge(
        client=FakeLlmClient(payload={"verdict": "maybe", "confidence": 0.5}),
        engine=engine,
        run_id=run.id,
        model="claude-sonnet-5-5",
    )
    grading = ModelGrading(awarded_item_ids=frozenset({"r1"}), total_points=1.0, feedback="x")

    assert await judge.assess(make_case("a"), grading) is None
