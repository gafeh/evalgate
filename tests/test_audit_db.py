import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.core.audit import result_fingerprint
from evalgate.db.audit import (
    AlreadySealedError,
    chain_length,
    load_chain,
    seal_run,
    verify_run,
)
from evalgate.db.models import AuditEntry, Result, Run
from evalgate.db.repositories import create_run, save_dataset
from evalgate.db.session import session_scope
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader
from tests.factories import make_case, make_dataset

# A clean StubGrader scores every case 1.0, so a tamper test has to move a
# value to something that is genuinely different.
TAMPERED_SCORE = 0.25


async def _graded_run(engine: AsyncEngine, size: int = 6) -> Run:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i:03d}") for i in range(size)])
        )
        run = await create_run(session, dataset=dataset, model="stub-v1")
    await worker_loop(engine, run_id=run.id, grader=StubGrader(), worker_id="w1")
    return run


async def test_sealing_a_run_verifies(engine: AsyncEngine) -> None:
    run = await _graded_run(engine)
    async with session_scope(engine) as session:
        verification = await seal_run(session, run.id)
        assert verification.ok is True
        assert verification.links == 6
        assert await chain_length(session, run.id) == 6
        assert (await verify_run(session, run.id)).ok is True


async def test_sealing_twice_is_refused(engine: AsyncEngine) -> None:
    run = await _graded_run(engine, 3)
    async with session_scope(engine) as session:
        await seal_run(session, run.id)
        with pytest.raises(AlreadySealedError, match="already has 3"):
            await seal_run(session, run.id)


async def test_resealing_replaces_the_chain(engine: AsyncEngine) -> None:
    run = await _graded_run(engine, 3)
    async with session_scope(engine) as session:
        first = await seal_run(session, run.id)
        again = await seal_run(session, run.id, reseal=True)
        assert await chain_length(session, run.id) == 3
        assert again.root == first.root


async def test_sealing_is_deterministic(engine: AsyncEngine) -> None:
    """Same results, same root. Otherwise the root proves nothing."""
    run = await _graded_run(engine, 5)
    async with session_scope(engine) as session:
        first = await seal_run(session, run.id)
        second = await seal_run(session, run.id, reseal=True)
    assert first.root == second.root


async def test_editing_a_result_is_caught(engine: AsyncEngine) -> None:
    run = await _graded_run(engine, 4)
    async with session_scope(engine) as session:
        await seal_run(session, run.id)

    async with session_scope(engine) as session:
        target = (
            await session.execute(select(Result.id).where(Result.run_id == run.id))
        ).scalars().first()
        await session.execute(
            update(Result).where(Result.id == target).values(overall=TAMPERED_SCORE)
        )
        await session.commit()

        verification = await verify_run(session, run.id)
        assert verification.ok is False
        assert any(problem.kind == "row" for problem in verification.problems)
        assert any(
            "own fingerprint" in str(problem) for problem in verification.problems
        )


async def test_a_consistent_cover_up_is_still_caught(engine: AsyncEngine) -> None:
    """Edit the row and its fingerprint together. The chain recorded the old one."""
    run = await _graded_run(engine, 4)
    async with session_scope(engine) as session:
        await seal_run(session, run.id)

    async with session_scope(engine) as session:
        result = (
            await session.execute(select(Result).where(Result.run_id == run.id))
        ).scalars().first()
        assert result is not None

        result.overall = TAMPERED_SCORE
        result.content_hash = result_fingerprint(
            run_id=str(result.run_id),
            case_id=str(result.case_id),
            raw_output=result.raw_output,
            grading=result.grading,
            scores=result.scores,
            overall=TAMPERED_SCORE,
        )
        await session.commit()

        verification = await verify_run(session, run.id)
        assert verification.ok is False
        assert any(
            "differs from what the chain recorded" in str(problem)
            for problem in verification.problems
        )


async def test_rewriting_the_chain_is_caught(engine: AsyncEngine) -> None:
    run = await _graded_run(engine, 4)
    async with session_scope(engine) as session:
        await seal_run(session, run.id)

    async with session_scope(engine) as session:
        entry = (
            await session.execute(
                select(AuditEntry).where(
                    AuditEntry.run_id == run.id, AuditEntry.seq == 2
                )
            )
        ).scalars().one()
        entry.payload = {**entry.payload, "overall": 9.9}
        await session.commit()

        verification = await verify_run(session, run.id)
        assert verification.ok is False
        assert any(problem.kind == "payload" for problem in verification.problems)


async def test_deleting_an_entry_is_caught_as_coverage(engine: AsyncEngine) -> None:
    run = await _graded_run(engine, 4)
    async with session_scope(engine) as session:
        await seal_run(session, run.id)

    async with session_scope(engine) as session:
        entry = (
            await session.execute(
                select(AuditEntry).where(
                    AuditEntry.run_id == run.id, AuditEntry.seq == 4
                )
            )
        ).scalars().one()
        await session.delete(entry)
        await session.commit()

        verification = await verify_run(session, run.id)
        assert verification.ok is False
        assert any(problem.kind == "coverage" for problem in verification.problems)


async def test_an_unsealed_run_has_an_empty_chain(engine: AsyncEngine) -> None:
    run = await _graded_run(engine, 2)
    async with session_scope(engine) as session:
        assert await load_chain(session, run.id) == []
        verification = await verify_run(session, run.id)
        assert verification.ok is False
        assert any(problem.kind == "coverage" for problem in verification.problems)