import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from evalgate.db.repositories import (
    DatasetConflictError,
    get_dataset,
    list_cases,
    save_dataset,
)
from tests.factories import make_case, make_dataset


async def test_save_dataset_stores_cases(session: AsyncSession) -> None:
    dataset = make_dataset(make_case("a"), make_case("b"))
    row = await save_dataset(session, dataset)

    assert row.case_count == 2
    assert row.content_hash == dataset.content_hash()
    cases = await list_cases(session, row.id)
    assert [case.external_id for case in cases] == ["a", "b"]
    assert cases[0].rubric["items"][1]["points"] == 2.0
    assert cases[0].tags["error_mode"] == "off_by_one"


async def test_save_dataset_is_idempotent(session: AsyncSession) -> None:
    dataset = make_dataset(make_case("a"))
    first = await save_dataset(session, dataset)
    second = await save_dataset(session, dataset)

    assert first.id == second.id
    assert len(await list_cases(session, first.id)) == 1


async def test_resaving_changed_content_is_rejected(session: AsyncSession) -> None:
    await save_dataset(session, make_dataset(make_case("a")))
    changed = make_dataset(make_case("a"), make_case("b"))

    with pytest.raises(DatasetConflictError, match="Bump the version"):
        await save_dataset(session, changed)


async def test_get_dataset_returns_none_when_absent(session: AsyncSession) -> None:
    assert await get_dataset(session, "nope", "1") is None