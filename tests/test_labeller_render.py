from uuid import uuid4

import pytest

from evalgate.db.models import ReviewReason
from evalgate.db.review import ReviewTask
from evalgate.labeller.render import parse_answer, render_task


def make_task(*, parsed_ok: bool = True, raw_output: str = "{}") -> ReviewTask:
    return ReviewTask(
        review_item_id=uuid4(),
        reason=ReviewReason.CALIBRATION,
        case_id="case-0007",
        question_prompt="Write sumAll(int[] values).",
        rubric_lines=("r1 (1 pt): initialises accumulator",),
        student_response="int total = 0;",
        expected_awards=("r1",),
        expected_points=1.0,
        grader_awards=("r1", "r2"),
        grader_points=3.0,
        grader_feedback="Nice work, but check your loop bound.",
        must_mention=("loop bound",),
        must_not_mention=("full solution",),
        parsed_ok=parsed_ok,
        raw_output=raw_output,
    )


def test_render_shows_what_a_labeller_needs() -> None:
    text = render_task(make_task(), position=3, remaining=12)
    assert "case-0007" in text
    assert "3 done, 12 left" in text
    assert "Nice work, but check your loop bound." in text
    assert "r1 (1 pt): initialises accumulator" in text
    assert "loop bound" in text


def test_render_never_mentions_confidence_or_a_judge_verdict() -> None:
    text = render_task(make_task(), position=0, remaining=1).lower()
    assert "confidence" not in text
    assert "judge" not in text


def test_render_surfaces_raw_output_when_parsing_failed() -> None:
    broken = make_task(parsed_ok=False, raw_output="Sure! Here goes:")
    text = render_task(broken, position=0, remaining=1)
    assert "did not parse" in text
    assert "Sure! Here goes:" in text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("p", "pass"),
        ("pass", "pass"),
        ("PASS", "pass"),
        ("  a  ", "partial"),
        ("partial", "partial"),
        ("f", "fail"),
        ("fail", "fail"),
        ("q", "quit"),
        ("quit", "quit"),
        ("exit", "quit"),
        ("s", "skip"),
        ("skip", "skip"),
        ("", None),
        ("yes", None),
        ("3", None),
    ],
)
def test_parse_answer_takes_keys_and_whole_words(
    text: str, expected: str | None
) -> None:
    assert parse_answer(text) == expected