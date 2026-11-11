from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.db.models import Run
from evalgate.db.repositories import create_run, save_dataset
from evalgate.db.session import session_scope
from evalgate.runner.llm_grader import LlmGrader
from tests.factories import make_case, make_dataset
from tests.fakes import FakeLlmClient


async def _a_run(engine: AsyncEngine) -> Run:
    async with session_scope(engine) as session:
        dataset = await save_dataset(session, make_dataset(make_case("a")))
        return await create_run(session, dataset=dataset, model="claude-haiku-5-5")


async def test_grader_builds_a_grading_from_the_tool_payload(engine: AsyncEngine) -> None:
    run = await _a_run(engine)
    client = FakeLlmClient(
        payload={
            "awarded_item_ids": ["r1", "r3"],
            "total_points": 2,
            "feedback": "Your loop runs one index too far.",
        }
    )
    grader = LlmGrader(
        client=client, engine=engine, run_id=run.id, model="claude-haiku-5-5"
    )

    output = await grader.grade(make_case("a"))

    assert output.parse_error is None
    assert output.grading is not None
    assert output.grading.awarded_item_ids == frozenset({"r1", "r3"})
    assert output.grading.total_points == 2.0


async def test_grader_reports_a_parse_error_when_no_tool_is_used(
    engine: AsyncEngine,
) -> None:
    run = await _a_run(engine)
    grader = LlmGrader(
        client=FakeLlmClient(payload=None),
        engine=engine,
        run_id=run.id,
        model="claude-haiku-5-5",
    )

    output = await grader.grade(make_case("a"))

    assert output.grading is None
    assert output.parse_error is not None
    assert output.raw_output


async def test_grader_rejects_a_payload_that_breaks_the_contract(
    engine: AsyncEngine,
) -> None:
    run = await _a_run(engine)
    grader = LlmGrader(
        client=FakeLlmClient(payload={"awarded_item_ids": ["r1"]}),
        engine=engine,
        run_id=run.id,
        model="claude-haiku-5-5",
    )

    output = await grader.grade(make_case("a"))

    assert output.grading is None
    assert "contract" in (output.parse_error or "")


async def test_grading_the_same_case_twice_costs_one_call(engine: AsyncEngine) -> None:
    run = await _a_run(engine)
    client = FakeLlmClient(
        payload={"awarded_item_ids": ["r1"], "total_points": 1, "feedback": "ok"}
    )
    grader = LlmGrader(
        client=client, engine=engine, run_id=run.id, model="claude-haiku-5-5"
    )

    await grader.grade(make_case("a"))
    await grader.grade(make_case("a"))

    assert client.call_count == 1