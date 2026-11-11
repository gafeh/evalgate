from evalgate.core.grading import ModelGrading, evaluate_grading
from evalgate.core.scorers import JsonSchemaValid
from tests.factories import make_case


def test_perfect_grading_scores_one() -> None:
    case = make_case()
    grading = ModelGrading(
        awarded_item_ids=case.expected.awarded_item_ids,
        total_points=case.expected_points,
        feedback="Your loop runs one index past the end of the array.",
    )
    result = evaluate_grading(case, grading)
    assert result.overall == 1.0
    assert result.exact_rubric_match is True


def test_over_credit_is_reported_separately_from_under_credit() -> None:
    case = make_case(awarded=frozenset({"r1"}))
    grading = ModelGrading(awarded_item_ids=frozenset({"r1", "r2"}), total_points=3.0)
    result = evaluate_grading(case, grading)
    assert result.over_credit_points == 2.0
    assert result.under_credit_points == 0.0
    assert result.exact_rubric_match is False


def test_under_credit_is_reported_separately() -> None:
    case = make_case(awarded=frozenset({"r1", "r2"}))
    grading = ModelGrading(awarded_item_ids=frozenset({"r1"}), total_points=1.0)
    result = evaluate_grading(case, grading)
    assert result.under_credit_points == 2.0
    assert result.over_credit_points == 0.0


def test_awarding_an_item_that_does_not_exist_is_flagged() -> None:
    case = make_case()
    grading = ModelGrading(awarded_item_ids=frozenset({"r1", "ghost"}), total_points=2.0)
    assert evaluate_grading(case, grading).hallucinated_item_ids == ("ghost",)


def test_point_tolerance_forgives_rounding_only() -> None:
    case = make_case(awarded=frozenset({"r2"}))
    grading = ModelGrading(awarded_item_ids=frozenset({"r2"}), total_points=2.05)
    assert evaluate_grading(case, grading, point_tolerance=0.1).overall == 1.0
    assert evaluate_grading(case, grading, point_tolerance=0.0).overall < 1.0


def test_model_output_contract_validates_against_its_own_schema() -> None:
    grading = ModelGrading(awarded_item_ids=frozenset({"r1"}), total_points=1.0)
    score = JsonSchemaValid().score(
        ModelGrading.output_schema(), grading.model_dump(mode="json")
    )
    assert score.passed is True