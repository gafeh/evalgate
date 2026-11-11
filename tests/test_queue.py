import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.db.models import Run, RunStatus, WorkItem, WorkItemStatus
from evalgate.db.queue import (
    claim_batch,
    complete_work_item,
    fail_work_item,
    finalize_run_if_complete,
    outstanding_count,
    reclaim_stale,
)
from evalgate.db.repositories import create_run, save_dataset, work_item_counts
from evalgate.db.session import session_scope
from tests.factories import make_case, make_dataset


async def _seed(engine: AsyncEngine, size: int) -> Run:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i:03d}") for i in range(size)])
        )
        return await create_run(session, dataset=dataset, model="stub-v1")


async def test_claim_respects_batch_size(engine: AsyncEngine) -> None:
    run = await _seed(engine, 10)
    async with session_scope(engine) as session:
        claims = await claim_batch(session, run_id=run.id, worker_id="w1", batch_size=4)
        assert len(claims) == 4
        assert await work_item_counts(session, run.id) == {"claimed": 4, "queued": 6}


async def test_two_workers_get_disjoint_batches(engine: AsyncEngine) -> None:
    run = await _seed(engine, 20)
    async with session_scope(engine) as a, session_scope(engine) as b:
        claims_a, claims_b = await asyncio.gather(
            claim_batch(a, run_id=run.id, worker_id="a", batch_size=10),
            claim_batch(b, run_id=run.id, worker_id="b", batch_size=10),
        )
    ids_a = {claim.work_item_id for claim in claims_a}
    ids_b = {claim.work_item_id for claim in claims_b}
    assert len(ids_a) + len(ids_b) == 20
    assert not ids_a & ids_b


async def test_skip_locked_does_not_block(engine: AsyncEngine) -> None:
    """A worker holding row locks must not stall another worker's claim."""
    run = await _seed(engine, 6)
    async with session_scope(engine) as holder:
        locked = (
            await holder.execute(
                select(WorkItem.id)
                .where(WorkItem.run_id == run.id)
                .limit(3)
                .with_for_update(skip_locked=True)
            )
        ).scalars().all()
        assert len(locked) == 3

        async with session_scope(engine) as other:
            claims = await asyncio.wait_for(
                claim_batch(other, run_id=run.id, worker_id="other", batch_size=6),
                timeout=5.0,
            )
        assert len(claims) == 3
        assert not {claim.work_item_id for claim in claims} & set(locked)


async def test_claimed_items_are_not_reclaimable_while_fresh(engine: AsyncEngine) -> None:
    run = await _seed(engine, 5)
    async with session_scope(engine) as session:
        await claim_batch(session, run_id=run.id, worker_id="w1", batch_size=5)
        assert await reclaim_stale(session, run_id=run.id, stale_after_seconds=60) == 0


async def test_stale_claims_are_returned_to_the_queue(engine: AsyncEngine) -> None:
    run = await _seed(engine, 5)
    async with session_scope(engine) as session:
        await claim_batch(session, run_id=run.id, worker_id="dead", batch_size=5)
        # Treat every claim as older than the cutoff.
        reclaimed = await reclaim_stale(session, run_id=run.id, stale_after_seconds=-1)
        assert reclaimed == 5
        assert await work_item_counts(session, run.id) == {"queued": 5}


async def test_items_out_of_attempts_are_buried_not_requeued(engine: AsyncEngine) -> None:
    run = await _seed(engine, 3)
    async with session_scope(engine) as session:
        for _ in range(3):
            claims = await claim_batch(
                session, run_id=run.id, worker_id="w1", batch_size=3
            )
            for claim in claims:
                await fail_work_item(
                    session, claim.work_item_id, error="boom", backoff_seconds=0
                )
        counts = await work_item_counts(session, run.id)
        assert counts == {"dead": 3}


async def test_failure_backoff_makes_items_temporarily_unclaimable(
    engine: AsyncEngine,
) -> None:
    run = await _seed(engine, 2)
    async with session_scope(engine) as session:
        claims = await claim_batch(session, run_id=run.id, worker_id="w1", batch_size=2)
        for claim in claims:
            status = await fail_work_item(
                session, claim.work_item_id, error="rate limited", backoff_seconds=300
            )
            assert status is WorkItemStatus.QUEUED

        assert await claim_batch(session, run_id=run.id, worker_id="w1", batch_size=2) == []
        assert await outstanding_count(session, run.id) == 2


async def test_run_finalises_only_when_nothing_is_outstanding(
    engine: AsyncEngine,
) -> None:
    run = await _seed(engine, 3)
    async with session_scope(engine) as session:
        claims = await claim_batch(session, run_id=run.id, worker_id="w1", batch_size=3)
        assert await finalize_run_if_complete(session, run.id) is False

        for claim in claims:
            await complete_work_item(session, claim.work_item_id)

        assert await finalize_run_if_complete(session, run.id) is True
        assert await finalize_run_if_complete(session, run.id) is False

        refreshed = await session.get(Run, run.id)
        assert refreshed is not None
        assert refreshed.status is RunStatus.COMPLETED
        assert refreshed.finished_at is not None