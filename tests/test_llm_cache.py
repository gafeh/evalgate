from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.db.models import CallPurpose, LlmInvocation, Run
from evalgate.db.repositories import create_run, save_dataset
from evalgate.db.session import session_scope
from evalgate.judge.cache import complete_with_cache, run_usage
from evalgate.judge.client import LlmRequest
from tests.factories import make_case, make_dataset
from tests.fakes import FakeLlmClient

REQUEST = LlmRequest(
    model="claude-haiku-5-5",
    system="be a grader",
    user="grade this",
    tool_name="submit_grading",
    tool_description="submit",
    input_schema={"type": "object"},
)


async def _a_run(engine: AsyncEngine) -> Run:
    async with session_scope(engine) as session:
        dataset = await save_dataset(session, make_dataset(make_case("a")))
        return await create_run(session, dataset=dataset, model="claude-haiku-5-5")


async def test_first_call_misses_then_second_call_hits(engine: AsyncEngine) -> None:
    run = await _a_run(engine)
    client = FakeLlmClient(payload={"total_points": 3})

    async with session_scope(engine) as session:
        first, cached_first = await complete_with_cache(
            session, client, REQUEST, purpose=CallPurpose.GRADER, run_id=run.id
        )
        second, cached_second = await complete_with_cache(
            session, client, REQUEST, purpose=CallPurpose.GRADER, run_id=run.id
        )

    assert cached_first is False
    assert cached_second is True
    assert client.call_count == 1
    assert first.payload == second.payload


async def test_cache_hits_are_free_and_the_ledger_says_so(engine: AsyncEngine) -> None:
    run = await _a_run(engine)
    client = FakeLlmClient(payload={"total_points": 3}, input_tokens=200, output_tokens=80)

    async with session_scope(engine) as session:
        for _ in range(3):
            await complete_with_cache(
                session, client, REQUEST, purpose=CallPurpose.GRADER, run_id=run.id
            )
        usage = await run_usage(session, run.id)

    assert client.call_count == 1
    assert usage.calls == 3
    assert usage.cache_hits == 2
    assert usage.input_tokens == 200
    assert usage.output_tokens == 80
    assert usage.cost_usd > 0.0
    assert usage.cache_hit_rate == 2 / 3


async def test_a_changed_prompt_is_a_different_cache_key(engine: AsyncEngine) -> None:
    run = await _a_run(engine)
    client = FakeLlmClient(payload={"total_points": 3})
    other = LlmRequest(
        model=REQUEST.model,
        system=REQUEST.system,
        user="grade this differently",
        tool_name=REQUEST.tool_name,
        tool_description=REQUEST.tool_description,
        input_schema=REQUEST.input_schema,
    )

    async with session_scope(engine) as session:
        await complete_with_cache(
            session, client, REQUEST, purpose=CallPurpose.GRADER, run_id=run.id
        )
        await complete_with_cache(
            session, client, other, purpose=CallPurpose.GRADER, run_id=run.id
        )
        keys = (await session.execute(select(LlmInvocation.cache_key))).scalars().all()

    assert client.call_count == 2
    assert len(set(keys)) == 2


async def test_max_tokens_is_part_of_the_cache_key() -> None:
    longer = LlmRequest(
        model=REQUEST.model,
        system=REQUEST.system,
        user=REQUEST.user,
        tool_name=REQUEST.tool_name,
        tool_description=REQUEST.tool_description,
        input_schema=REQUEST.input_schema,
        max_tokens=4096,
    )
    assert REQUEST.cache_key() != longer.cache_key()