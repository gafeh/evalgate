from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evalgate.core.hashing import content_hash

GENESIS = "0" * 64


def result_fingerprint(
    *,
    run_id: str,
    case_id: str,
    raw_output: str,
    grading: Mapping[str, Any] | None,
    scores: Mapping[str, Any],
    overall: float | None,
) -> str:
    """Fingerprint of one scored result.

    Both the writer and the verifier compute this from the same fields, so an
    edit to any of them is detectable. Kept in core so there is exactly one
    definition of what a result's identity is.
    """
    return content_hash(
        {
            "run_id": run_id,
            "case_id": case_id,
            "raw_output": raw_output,
            "grading": dict(grading) if grading is not None else None,
            "scores": dict(scores),
            "overall": overall,
        }
    )


def link_hash(*, seq: int, prev_hash: str, payload: Mapping[str, Any]) -> str:
    return content_hash({"seq": seq, "prev": prev_hash, "payload": dict(payload)})


@dataclass(frozen=True, slots=True)
class ChainLink:
    seq: int
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str


def build_chain(
    payloads: Sequence[Mapping[str, Any]], *, genesis: str = GENESIS
) -> tuple[ChainLink, ...]:
    """Hash-chain a sequence of payloads. Each link covers everything before it."""
    links: list[ChainLink] = []
    previous = genesis
    for index, payload in enumerate(payloads, start=1):
        digest = link_hash(seq=index, prev_hash=previous, payload=payload)
        links.append(
            ChainLink(
                seq=index,
                payload=dict(payload),
                prev_hash=previous,
                entry_hash=digest,
            )
        )
        previous = digest
    return tuple(links)


@dataclass(frozen=True, slots=True)
class ChainProblem:
    seq: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"seq {self.seq}: {self.kind}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Verification:
    links: int
    root: str | None
    problems: tuple[ChainProblem, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


def verify_chain(
    links: Sequence[ChainLink], *, genesis: str = GENESIS
) -> Verification:
    """Check sequence, linkage and each link's own hash.

    Rewriting one payload breaks that link's hash. Rewriting the payload and
    its hash breaks the next link's prev_hash. Rewriting the whole chain
    forward still changes the root, which is why the root goes in the report.
    """
    problems: list[ChainProblem] = []
    previous = genesis

    for index, link in enumerate(links, start=1):
        if link.seq != index:
            problems.append(
                ChainProblem(link.seq, "sequence", f"expected seq {index}")
            )
        if link.prev_hash != previous:
            problems.append(
                ChainProblem(
                    link.seq,
                    "linkage",
                    f"prev_hash {link.prev_hash[:12]} does not follow "
                    f"{previous[:12]}",
                )
            )
        expected = link_hash(
            seq=link.seq, prev_hash=link.prev_hash, payload=link.payload
        )
        if link.entry_hash != expected:
            problems.append(
                ChainProblem(
                    link.seq,
                    "payload",
                    "entry_hash does not match the payload it covers",
                )
            )
        previous = link.entry_hash

    return Verification(
        links=len(links),
        root=links[-1].entry_hash if links else None,
        problems=tuple(problems),
    )