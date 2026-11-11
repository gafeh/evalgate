from evalgate.core.dataset import Dataset, ExpectedGrading, GradingCase, Rubric, RubricItem

RUBRIC = Rubric(
    items=(
        RubricItem(id="r1", description="Declares and initialises the accumulator", points=1.0),
        RubricItem(id="r2", description="Loops over every element exactly once", points=2.0),
        RubricItem(id="r3", description="Returns the accumulated value", points=1.0),
    )
)


def make_case(
    case_id: str = "c1",
    awarded: frozenset[str] = frozenset({"r1", "r3"}),
    tags: dict[str, str] | None = None,
) -> GradingCase:
    return GradingCase(
        id=case_id,
        question_id="2026-frq-1a",
        question_prompt="Write a method sumAll(int[] values) that returns the sum.",
        rubric=RUBRIC,
        student_response="int total = 0; for (int i = 0; i <= values.length; i++) "
        "{ total += values[i]; } return total;",
        expected=ExpectedGrading(
            awarded_item_ids=awarded,
            notes="Off-by-one in the loop bound, so the traversal point is lost.",
            must_mention=("loop bound",),
            must_not_mention=("perfect",),
        ),
        tags=tags or {"error_mode": "off_by_one", "unit": "arrays", "difficulty": "medium"},
    )


def make_dataset(*cases: GradingCase) -> Dataset:
    return Dataset(
        name="apcs-frq-smoke",
        version="1",
        cases=cases or (make_case(),),
    )