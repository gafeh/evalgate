from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.core.judgment import CRITERIA, JudgeVerdict
from evalgate.db.models import Run
from evalgate.db.repositories import (
    create_run,
    get_judge_assessment,
    list_results,
    save_dataset,
    work_item_counts,
)
from evalgate.db.session import session_scope
from evalgate.judge.assessor import LlmJudge
from evalgate.judge.cache import run_usage
from evalgate.runner.llm_grader import LlmGrader
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader
from tests.factories import make_case, make_dataset
from tests.fakes import FakeLlmClient

GRADER_PAYLOAD = {
    "awarded_item_ids": ["r1", "r3"],
    "total_points": 2,
    "feedback": "Your loop runs one index past the end.",
}
JUDGE_PAYLOAD = {
    "reasoning": "Names the real defect, no code given.",
    "verdict": "pass",
    "confidence": 0.77,
    "criteria": dict.fromkeys(CRITERIA, True),
}


async def _a_run(engine: AsyncEngine, size: int) -> Run:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i:03d}") for i in range(size)])
        )
        return await create_run(session, dataset=dataset, model="claude-haiku-5-5")


async def test_full_pass_grades_judges_and_accounts(engine: AsyncEngine) -> None:
    run = await _a_run(engine, 4)
    client = FakeLlmClient(
        payload_by_tool={
            "submit_grading": GRADER_PAYLOAD,
            "submit_judgment": JUDGE_PAYLOAD,
        }
    )
    grader = LlmGrader(
        client=client, engine=engine, run_id=run.id, model="claude-haiku-5-5"
    )
    judge = LlmJudge(
        client=client, engine=engine, run_id=run.id, model="claude-sonnet-5-5"
    )

    stats = await worker_loop(
        engine, run_id=run.id, grader=grader, worker_id="w1", batch_size=2, judge=judge
    )

    assert stats.succeeded == 4
    assert stats.judged == 4
    assert stats.judge_failures == 0

    async with session_scope(engine) as session:
        assert await work_item_counts(session, run.id) == {"done": 4}
        results = await list_results(session, run.id)
        assert len(results) == 4
        for result in results:
            assessment = await get_judge_assessment(session, result.id)
            assert assessment is not None
            assert assessment.verdict is JudgeVerdict.PASS
            assert assessment.confidence == 0.77

        usage = await run_usage(session, run.id)
        assert usage.calls == 8
        assert usage.cost_usd > 0.0


async def test_a_broken_judge_does_not_fail_the_run(engine: AsyncEngine) -> None:
    run = await _a_run(engine, 3)
    judge = LlmJudge(
        client=FakeLlmClient(raise_on_call=RuntimeError("judge is down")),
        engine=engine,
        run_id=run.id,
        model="claude-sonnet-5-5",
    )

    stats = await worker_loop(
        engine,
        run_id=run.id,
        grader=StubGrader(),
        worker_id="w1",
        batch_size=3,
        judge=judge,
    )

    assert stats.succeeded == 3
    assert stats.judged == 0
    assert stats.judge_failures == 3
    async with session_scope(engine) as session:
        assert await work_item_counts(session, run.id) == {"done": 3}
