import pytest

from evalgate.core.scorers import JsonSchemaValid, SetF1


@pytest.mark.parametrize(
    ("expected", "actual", "want_f1"),
    [
        ({"a", "b"}, {"a", "b"}, 1.0),
        (set(), set(), 1.0),
        ({"a", "b"}, {"a"}, pytest.approx(2 / 3)),
        ({"a"}, {"a", "b"}, pytest.approx(2 / 3)),
        ({"a"}, {"b"}, 0.0),
        ({"a"}, set(), 0.0),
    ],
)
def test_set_f1_values(expected: set[str], actual: set[str], want_f1: float) -> None:
    assert SetF1().score(expected, actual).value == want_f1


def test_set_f1_reports_both_error_directions() -> None:
    detail = SetF1().score({"a", "b"}, {"b", "c"}).detail
    assert detail["false_positives"] == ["c"]
    assert detail["false_negatives"] == ["a"]


def test_set_f1_threshold_allows_partial_pass() -> None:
    assert SetF1(pass_threshold=0.6).score({"a", "b"}, {"a"}).passed is True


SCHEMA = {
    "type": "object",
    "required": ["total_points"],
    "properties": {"total_points": {"type": "number"}},
    "additionalProperties": False,
}


def test_json_schema_accepts_valid_payload() -> None:
    assert JsonSchemaValid().score(SCHEMA, {"total_points": 3}).passed is True


def test_json_schema_parses_json_strings() -> None:
    assert JsonSchemaValid().score(SCHEMA, '{"total_points": 3}').passed is True


def test_json_schema_rejects_malformed_json() -> None:
    score = JsonSchemaValid().score(SCHEMA, "{not json")
    assert score.passed is False
    assert "not valid json" in score.detail["errors"][0]


def test_json_schema_rejects_wrong_type() -> None:
    score = JsonSchemaValid().score(SCHEMA, {"total_points": "three"})
    assert score.passed is False
    assert score.detail["error_count"] == 1


def test_json_schema_rejects_extra_keys() -> None:
    assert JsonSchemaValid().score(SCHEMA, {"total_points": 3, "x": 1}).passed is False