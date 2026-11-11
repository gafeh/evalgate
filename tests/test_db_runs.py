from sqlalchemy.ext.asyncio import AsyncSession

from evalgate.db.models import RunStatus
from evalgate.db.repositories import (
    create_run,
    get_run,
    save_dataset,
    work_item_counts,
)
from tests.factories import make_case, make_dataset


async def test_create_run_enqueues_one_item_per_case(session: AsyncSession) -> None:
    dataset = await save_dataset(session, make_dataset(make_case("a"), make_case("b")))
    run = await create_run(session, dataset=dataset, model="claude-test", label="baseline")

    assert run.status is RunStatus.PENDING
    assert await work_item_counts(session, run.id) == {"queued": 2}


async def test_run_pins_the_dataset_content_hash(session: AsyncSession) -> None:
    core = make_dataset(make_case("a"))
    dataset = await save_dataset(session, core)
    run = await create_run(session, dataset=dataset, model="claude-test")

    assert run.dataset_content_hash == core.content_hash()


async def test_get_run_round_trips_params(session: AsyncSession) -> None:
    dataset = await save_dataset(session, make_dataset(make_case("a")))
    created = await create_run(
        session, dataset=dataset, model="claude-test", params={"temperature": 0.0}
    )
    fetched = await get_run(session, created.id)

    assert fetched is not None
    assert fetched.params == {"temperature": 0.0}