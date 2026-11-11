from evalgate.core.aggregate import EMPTY_TAG, summarize, summarize_by_slice
from evalgate.core.outcome import CaseOutcome


def outcome(
    case_id: str,
    overall: float | None = 1.0,
    *,
    parsed_ok: bool = True,
    over: float = 0.0,
    under: float = 0.0,
    mode: str | None = "off_by_one",
) -> CaseOutcome:
    return CaseOutcome(
        case_id=case_id,
        parsed_ok=parsed_ok,
        overall=overall,
        over_credit_points=over,
        under_credit_points=under,
        exact_match=parsed_ok and over == 0.0 and under == 0.0,
        tags={} if mode is None else {"error_mode": mode},
    )


def test_empty_summary_is_all_zero() -> None:
    summary = summarize([])
    assert summary.cases == 0
    assert summary.mean_overall == 0.0
    assert summary.unparseable_rate == 0.0


def test_unparseable_output_scores_zero_and_is_counted() -> None:
    summary = summarize([outcome("a", 1.0), outcome("b", None, parsed_ok=False)])
    assert summary.cases == 2
    assert summary.unparseable == 1
    assert summary.unparseable_rate == 0.5
    assert summary.mean_overall == 0.5


def test_credit_errors_are_tracked_in_both_directions() -> None:
    summary = summarize(
        [outcome("a", 0.6, over=2.0), outcome("b", 0.7, under=1.0), outcome("c", 1.0)]
    )
    assert summary.over_credit_points == 2.0
    assert summary.under_credit_points == 1.0
    assert summary.mean_over_credit == 2.0 / 3.0
    assert summary.exact_match_rate == 1 / 3


def test_slices_split_by_tag_value() -> None:
    slices = summarize_by_slice(
        [
            outcome("a", 1.0, mode="off_by_one"),
            outcome("b", 0.0, mode="off_by_one"),
            outcome("c", 1.0, mode="logic"),
        ],
        "error_mode",
    )
    assert set(slices) == {"off_by_one", "logic"}
    assert slices["off_by_one"].mean_overall == 0.5
    assert slices["logic"].cases == 1


def test_untagged_cases_get_their_own_bucket() -> None:
    slices = summarize_by_slice([outcome("a", 1.0, mode=None)], "error_mode")
    assert EMPTY_TAG in slices