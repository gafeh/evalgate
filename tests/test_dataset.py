import pytest
from pydantic import ValidationError

from evalgate.core.dataset import Rubric, RubricItem
from tests.factories import RUBRIC, make_case, make_dataset


def test_rubric_max_points_sums_items() -> None:
    assert RUBRIC.max_points == 4.0


def test_rubric_rejects_duplicate_item_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Rubric(items=(RubricItem(id="r1", description="a"), RubricItem(id="r1", description="b")))


def test_rubric_points_for_subset() -> None:
    assert RUBRIC.points_for(frozenset({"r1", "r2"})) == 3.0


def test_expected_points_uses_rubric_weights() -> None:
    case = make_case(awarded=frozenset({"r2"}))
    assert case.expected_points == 2.0


def test_case_rejects_awards_outside_its_rubric() -> None:
    with pytest.raises(ValidationError, match="unknown rubric items"):
        make_case(awarded=frozenset({"r1", "r9"}))


def test_dataset_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        make_dataset(make_case("dup"), make_case("dup"))


def test_dataset_hash_is_stable_and_content_sensitive() -> None:
    first = make_dataset(make_case("a"), make_case("b"))
    same = make_dataset(make_case("a"), make_case("b"))
    other = make_dataset(make_case("a"), make_case("c"))
    assert first.content_hash() == same.content_hash()
    assert first.content_hash() != other.content_hash()


def test_dataset_exposes_slice_values() -> None:
    dataset = make_dataset(
        make_case("a", tags={"error_mode": "off_by_one"}),
        make_case("b", tags={"error_mode": "logic"}),
    )
    assert dataset.slice_values("error_mode") == frozenset({"off_by_one", "logic"})