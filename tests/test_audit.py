import itertools

import pytest

from evalgate.core.audit import (
    GENESIS,
    ChainLink,
    build_chain,
    link_hash,
    result_fingerprint,
    verify_chain,
)

PAYLOADS = [{"case_id": f"c{index}", "overall": index / 10} for index in range(5)]


def test_an_empty_chain_verifies_with_no_root() -> None:
    verification = verify_chain([])
    assert verification.ok is True
    assert verification.links == 0
    assert verification.root is None


def test_a_built_chain_verifies() -> None:
    links = build_chain(PAYLOADS)
    assert len(links) == 5
    assert links[0].prev_hash == GENESIS
    assert verify_chain(links).ok is True


def test_each_link_points_at_the_one_before_it() -> None:
    links = build_chain(PAYLOADS)
    for earlier, later in itertools.pairwise(links):
        assert later.prev_hash == earlier.entry_hash


def test_the_root_depends_on_every_payload() -> None:
    first = build_chain(PAYLOADS)[-1].entry_hash
    changed = list(PAYLOADS)
    changed[0] = {**changed[0], "overall": 0.99}
    assert build_chain(changed)[-1].entry_hash != first


def test_order_matters() -> None:
    forward = build_chain(PAYLOADS)[-1].entry_hash
    backward = build_chain(list(reversed(PAYLOADS)))[-1].entry_hash
    assert forward != backward


def test_rewriting_a_payload_breaks_that_link() -> None:
    links = list(build_chain(PAYLOADS))
    links[2] = ChainLink(
        seq=links[2].seq,
        payload={**links[2].payload, "overall": 9.9},
        prev_hash=links[2].prev_hash,
        entry_hash=links[2].entry_hash,
    )
    verification = verify_chain(links)
    assert verification.ok is False
    assert any(problem.kind == "payload" for problem in verification.problems)


def test_rewriting_a_payload_and_its_hash_breaks_the_next_link() -> None:
    """The cover-up is what the chain is for."""
    links = list(build_chain(PAYLOADS))
    tampered_payload = {**links[2].payload, "overall": 9.9}
    links[2] = ChainLink(
        seq=3,
        payload=tampered_payload,
        prev_hash=links[2].prev_hash,
        entry_hash=link_hash(
            seq=3, prev_hash=links[2].prev_hash, payload=tampered_payload
        ),
    )
    verification = verify_chain(links)
    assert verification.ok is False
    assert any(problem.kind == "linkage" for problem in verification.problems)


def test_dropping_a_link_is_detected() -> None:
    links = list(build_chain(PAYLOADS))
    del links[2]
    verification = verify_chain(links)
    assert verification.ok is False
    assert {problem.kind for problem in verification.problems} & {
        "sequence",
        "linkage",
    }


def test_fingerprint_covers_every_field() -> None:
    base = {
        "run_id": "r",
        "case_id": "c",
        "raw_output": "{}",
        "grading": {"total_points": 3},
        "scores": {"set_f1": {"value": 1.0}},
        "overall": 1.0,
    }
    original = result_fingerprint(**base)  # type: ignore[arg-type]
    for key, value in [
        ("raw_output", "tampered"),
        ("grading", {"total_points": 4}),
        ("scores", {"set_f1": {"value": 0.5}}),
        ("overall", 0.5),
        ("case_id", "other"),
    ]:
        changed = {**base, key: value}
        assert result_fingerprint(**changed) != original, key  # type: ignore[arg-type]


def test_fingerprint_is_stable() -> None:
    args = {
        "run_id": "r",
        "case_id": "c",
        "raw_output": "{}",
        "grading": None,
        "scores": {},
        "overall": None,
    }
    assert result_fingerprint(**args) == result_fingerprint(**args)  # type: ignore[arg-type]


def test_a_problem_renders_readably() -> None:
    links = list(build_chain(PAYLOADS))
    links[1] = ChainLink(1, {}, links[1].prev_hash, links[1].entry_hash)
    text = str(verify_chain(links).problems[0])
    assert "seq" in text
    assert ":" in text


def test_verification_rejects_a_bad_genesis() -> None:
    links = build_chain(PAYLOADS)
    verification = verify_chain(links, genesis="f" * 64)
    assert verification.ok is False
    assert verification.problems[0].kind == "linkage"


@pytest.mark.parametrize("size", [0, 1, 2, 50])
def test_chains_of_any_length_verify(size: int) -> None:
    payloads = [{"i": index} for index in range(size)]
    assert verify_chain(build_chain(payloads)).ok is True