import pytest

from evalgate.core.scorers import get_scorer_class, registered_scorers
from evalgate.core.types import Score


def test_score_rejects_out_of_range_value() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Score(scorer="x", value=1.5, passed=True)


def test_score_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Score(scorer="", value=1.0, passed=True)


def test_all_six_scorers_are_registered() -> None:
    assert registered_scorers() == (
        "exact_match",
        "json_schema_valid",
        "normalized_match",
        "numeric_tolerance",
        "regex_match",
        "set_f1",
    )


def test_unknown_scorer_lookup_is_loud() -> None:
    with pytest.raises(KeyError, match="unknown scorer"):
        get_scorer_class("vibes")