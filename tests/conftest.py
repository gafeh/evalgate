import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import evalgate.db.models  # noqa: F401  populates Base.metadata
from evalgate.config import get_settings
from evalgate.db.base import Base

# Every table, so a cached LLM response from one test cannot answer another.
TRUNCATE_ALL = (
    "TRUNCATE human_labels, review_items, judge_assessments, llm_calls, "
    "llm_invocations, results, work_items, runs, cases, datasets CASCADE"
)

@pytest.fixture(scope="session", autouse=True)
def _schema() -> Iterator[None]:
    """Rebuild the test schema once per test session."""

    async def rebuild() -> None:
        engine = create_async_engine(get_settings().test_database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(rebuild())
    yield


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A session inside an outer transaction that is always rolled back.

    join_transaction_mode="create_savepoint" means repository code can call
    commit() normally and still leave the database untouched after the test.
    """
    engine = create_async_engine(get_settings().test_database_url)
    conn = await engine.connect()
    outer = await conn.begin()
    factory = async_sessionmaker(
        bind=conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    db = factory()
    try:
        yield db
    finally:
        await db.close()
        await outer.rollback()
        await conn.close()
        await engine.dispose()


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """A real engine whose writes commit. Every table is truncated afterwards."""
    eng = create_async_engine(get_settings().test_database_url)
    try:
        yield eng
    finally:
        async with eng.begin() as conn:
            await conn.execute(text(TRUNCATE_ALL))
        await eng.dispose()