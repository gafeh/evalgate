from evalgate.core.hashing import canonical_json, content_hash


def test_key_order_does_not_change_the_hash() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_sets_hash_independently_of_iteration_order() -> None:
    assert content_hash({"s": {"b", "a", "c"}}) == content_hash({"s": {"c", "a", "b"}})


def test_different_content_gives_a_different_hash() -> None:
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_canonical_json_has_no_padding() -> None:
    assert canonical_json({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'