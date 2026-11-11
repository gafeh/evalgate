import asyncio

from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.db.models import Run, RunStatus
from evalgate.db.queue import claim_batch
from evalgate.db.repositories import (
    create_run,
    list_results,
    save_dataset,
    work_item_counts,
)
from evalgate.db.session import session_scope
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader
from tests.factories import make_case, make_dataset


async def _seed(engine: AsyncEngine, size: int) -> Run:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i:03d}") for i in range(size)])
        )
        return await create_run(session, dataset=dataset, model="stub-v1")


async def test_single_worker_completes_the_run(engine: AsyncEngine) -> None:
    run = await _seed(engine, 12)
    stats = await worker_loop(
        engine, run_id=run.id, grader=StubGrader(), worker_id="w1", batch_size=5
    )

    assert stats.succeeded == 12
    assert stats.duplicates == 0
    async with session_scope(engine) as session:
        assert await work_item_counts(session, run.id) == {"done": 12}
        assert len(await list_results(session, run.id)) == 12
        refreshed = await session.get(Run, run.id)
        assert refreshed is not None
        assert refreshed.status is RunStatus.COMPLETED


async def test_two_workers_split_the_run_without_double_grading(
    engine: AsyncEngine,
) -> None:
    run = await _seed(engine, 40)
    first, second = await asyncio.gather(
        worker_loop(
            engine, run_id=run.id, grader=StubGrader(), worker_id="w1", batch_size=4
        ),
        worker_loop(
            engine, run_id=run.id, grader=StubGrader(), worker_id="w2", batch_size=4
        ),
    )

    assert first.succeeded + second.succeeded == 40
    assert first.succeeded > 0
    assert second.succeeded > 0
    async with session_scope(engine) as session:
        assert len(await list_results(session, run.id)) == 40


async def test_a_dead_workers_claims_get_finished_by_someone_else(
    engine: AsyncEngine,
) -> None:
    """The crash scenario: claims taken, process gone, nothing recorded."""
    run = await _seed(engine, 10)
    async with session_scope(engine) as session:
        orphaned = await claim_batch(
            session, run_id=run.id, worker_id="crashed", batch_size=10
        )
    assert len(orphaned) == 10

    stats = await worker_loop(
        engine,
        run_id=run.id,
        grader=StubGrader(),
        worker_id="survivor",
        batch_size=5,
        stale_after_seconds=-1,
        poll_interval=0.01,
    )

    assert stats.reclaimed == 10
    assert stats.succeeded == 10
    async with session_scope(engine) as session:
        assert await work_item_counts(session, run.id) == {"done": 10}
        assert len(await list_results(session, run.id)) == 10


async def test_grader_exceptions_retry_then_bury(engine: AsyncEngine) -> None:
    run = await _seed(engine, 6)
    stats = await worker_loop(
        engine,
        run_id=run.id,
        grader=StubGrader(raise_rate=1.0),
        worker_id="w1",
        batch_size=6,
        poll_interval=0.01,
        idle_limit=1,
    )

    assert stats.succeeded == 0
    assert stats.failed >= 6
    async with session_scope(engine) as session:
        counts = await work_item_counts(session, run.id)
        assert counts.get("dead", 0) + counts.get("queued", 0) == 6


async def test_malformed_output_is_recorded_and_the_item_completes(
    engine: AsyncEngine,
) -> None:
    run = await _seed(engine, 8)
    await worker_loop(
        engine,
        run_id=run.id,
        grader=StubGrader(malformed_rate=1.0),
        worker_id="w1",
        batch_size=8,
    )

    async with session_scope(engine) as session:
        results = await list_results(session, run.id)
        assert len(results) == 8
        assert all(result.parsed_ok is False for result in results)
        assert all(result.overall is None for result in results)
        assert await work_item_counts(session, run.id) == {"done": 8}