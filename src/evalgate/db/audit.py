from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalgate.core.audit import (
    ChainLink,
    ChainProblem,
    Verification,
    build_chain,
    result_fingerprint,
    verify_chain,
)
from evalgate.db.models import AuditEntry, Case, Result

ENTRY_TYPE = "result"


class AlreadySealedError(RuntimeError):
    """This run already has a chain. Sealing twice would rewrite history."""


async def _result_payloads(
    session: AsyncSession, run_id: UUID
) -> list[dict[str, Any]]:
    """Results in a deterministic order, reduced to what the chain covers."""
    stmt = (
        select(Result, Case.external_id)
        .join(Case, Case.id == Result.case_id)
        .where(Result.run_id == run_id)
        .order_by(Case.external_id)
    )
    rows = (await session.execute(stmt)).all()
    return [
        {
            "case_id": external_id,
            "result_id": str(result.id),
            "fingerprint": result.content_hash,
            "overall": result.overall,
            "over_credit_points": result.over_credit_points,
            "under_credit_points": result.under_credit_points,
            "parsed_ok": result.parsed_ok,
        }
        for result, external_id in rows
    ]


async def chain_length(session: AsyncSession, run_id: UUID) -> int:
    stmt = select(AuditEntry.id).where(AuditEntry.run_id == run_id)
    return len((await session.execute(stmt)).scalars().all())


async def seal_run(
    session: AsyncSession, run_id: UUID, *, reseal: bool = False
) -> Verification:
    """Write the hash chain for a finished run. Returns its verification."""
    existing = await chain_length(session, run_id)
    if existing and not reseal:
        raise AlreadySealedError(
            f"run {run_id} already has {existing} audit entries; pass reseal to replace"
        )

    if existing:
        for entry in (
            await session.execute(
                select(AuditEntry).where(AuditEntry.run_id == run_id)
            )
        ).scalars():
            await session.delete(entry)
        await session.flush()

    links = build_chain(await _result_payloads(session, run_id))
    session.add_all(
        [
            AuditEntry(
                run_id=run_id,
                seq=link.seq,
                entry_type=ENTRY_TYPE,
                payload=link.payload,
                prev_hash=link.prev_hash,
                entry_hash=link.entry_hash,
            )
            for link in links
        ]
    )
    await session.commit()
    return verify_chain(links)


async def load_chain(session: AsyncSession, run_id: UUID) -> list[ChainLink]:
    stmt = (
        select(AuditEntry).where(AuditEntry.run_id == run_id).order_by(AuditEntry.seq)
    )
    return [
        ChainLink(
            seq=entry.seq,
            payload=dict(entry.payload),
            prev_hash=entry.prev_hash,
            entry_hash=entry.entry_hash,
        )
        for entry in (await session.execute(stmt)).scalars()
    ]


async def verify_run(session: AsyncSession, run_id: UUID) -> Verification:
    """Verify the chain, then verify the rows the chain claims to cover.

    Three separate failures are detectable. A rewritten result row no longer
    matches its own fingerprint. A result row rewritten together with its
    fingerprint no longer matches what the chain recorded. A rewritten chain
    fails the chain check itself.
    """
    links = await load_chain(session, run_id)
    verification = verify_chain(links)
    problems = list(verification.problems)

    stmt = (
        select(Result, Case.external_id)
        .join(Case, Case.id == Result.case_id)
        .where(Result.run_id == run_id)
        .order_by(Case.external_id)
    )
    rows = (await session.execute(stmt)).all()

    if len(rows) != len(links):
        problems.append(
            ChainProblem(
                0,
                "coverage",
                f"{len(rows)} results against {len(links)} audit entries",
            )
        )

    by_case = {link.payload.get("case_id"): link for link in links}
    for result, external_id in rows:
        link = by_case.get(external_id)
        if link is None:
            problems.append(
                ChainProblem(0, "coverage", f"case {external_id} is not in the chain")
            )
            continue

        recomputed = result_fingerprint(
            run_id=str(result.run_id),
            case_id=str(result.case_id),
            raw_output=result.raw_output,
            grading=result.grading,
            scores=result.scores,
            overall=result.overall,
        )
        if recomputed != result.content_hash:
            problems.append(
                ChainProblem(
                    link.seq,
                    "row",
                    f"case {external_id} no longer matches its own fingerprint",
                )
            )
        elif result.content_hash != link.payload.get("fingerprint"):
            problems.append(
                ChainProblem(
                    link.seq,
                    "row",
                    f"case {external_id} differs from what the chain recorded",
                )
            )

    return Verification(
        links=verification.links,
        root=verification.root,
        problems=tuple(problems),
    )