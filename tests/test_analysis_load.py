from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.analysis.load import load_outcomes
from evalgate.db.repositories import create_run, save_dataset
from evalgate.db.session import session_scope
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader
from tests.factories import make_case, make_dataset


async def test_outcomes_round_trip_from_a_real_run(engine: AsyncEngine) -> None:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i:03d}") for i in range(6)])
        )
        run = await create_run(session, dataset=dataset, model="stub-v1")

    await worker_loop(engine, run_id=run.id, grader=StubGrader(), worker_id="w1")

    async with session_scope(engine) as session:
        outcomes = await load_outcomes(session, run.id)

    assert len(outcomes) == 6
    assert [o.case_id for o in outcomes] == [f"c{i:03d}" for i in range(6)]
    assert all(o.parsed_ok for o in outcomes)
    assert all(o.exact_match for o in outcomes)
    assert all(o.tags["error_mode"] == "off_by_one" for o in outcomes)


async def test_over_crediting_shows_up_in_the_outcomes(engine: AsyncEngine) -> None:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i:03d}") for i in range(4)])
        )
        run = await create_run(session, dataset=dataset, model="stub-v1")

    await worker_loop(
        engine,
        run_id=run.id,
        grader=StubGrader(over_credit_rate=1.0),
        worker_id="w1",
    )

    async with session_scope(engine) as session:
        outcomes = await load_outcomes(session, run.id)

    assert all(o.over_credit_points > 0.0 for o in outcomes)
    assert not any(o.exact_match for o in outcomes)


async def test_unparseable_results_load_as_unscored(engine: AsyncEngine) -> None:
    async with session_scope(engine) as session:
        dataset = await save_dataset(session, make_dataset(make_case("a")))
        run = await create_run(session, dataset=dataset, model="stub-v1")

    await worker_loop(
        engine, run_id=run.id, grader=StubGrader(malformed_rate=1.0), worker_id="w1"
    )

    async with session_scope(engine) as session:
        outcomes = await load_outcomes(session, run.id)

    assert len(outcomes) == 1
    assert outcomes[0].parsed_ok is False
    assert outcomes[0].overall is None
    assert outcomes[0].effective_overall == 0.0