from __future__ import annotations

from typing import Any

from evalgate.core.dataset import GradingCase
from evalgate.core.grading import ModelGrading
from evalgate.core.judgment import CRITERIA

GRADER_TOOL_NAME = "submit_grading"
GRADER_TOOL_DESCRIPTION = (
    "Submit the rubric items earned by this student response, the point total, "
    "and short feedback addressed to the student."
)
GRADER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "awarded_item_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ids of the rubric items this response earns. May be empty.",
        },
        "total_points": {
            "type": "number",
            "description": "Sum of the points for the awarded items.",
        },
        "feedback": {
            "type": "string",
            "description": "Two or three sentences for the student. Do not give the solution.",
        },
    },
    "required": ["awarded_item_ids", "total_points", "feedback"],
}

GRADER_SYSTEM = (
    "You grade AP Computer Science A free-response questions against an official "
    "rubric. Award a rubric item only when the response clearly earns it. Partial "
    "credit comes from awarding some items, never from inventing fractional points. "
    "Never award an item that is not in the rubric you were given."
)

JUDGE_TOOL_NAME = "submit_judgment"
JUDGE_TOOL_DESCRIPTION = (
    "Judge whether the grader's feedback to the student is acceptable, given the "
    "rubric and the correct awards."
)
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Fill this in first: two or three sentences of analysis.",
        },
        "verdict": {"type": "string", "enum": ["pass", "partial", "fail"]},
        "confidence": {
            "type": "number",
            "description": (
                "How likely it is that a careful human grader would agree with this "
                "verdict, from 0 to 1."
            ),
        },
        "criteria": {
            "type": "object",
            "properties": {name: {"type": "boolean"} for name in CRITERIA},
            "required": list(CRITERIA),
        },
    },
    "required": ["reasoning", "verdict", "confidence", "criteria"],
}

JUDGE_SYSTEM = (
    "You review feedback written by an automated grader for an AP Computer Science "
    "student. You are given the question, the rubric, the student's response, the "
    "correct rubric awards, and the grader's feedback.\n\n"
    "Judge the feedback only. Criteria:\n"
    "- accurate: it does not misstate what the student did\n"
    "- actionable: the student could act on it\n"
    "- covers_required_points: it raises the specific issues listed as required\n"
    "- no_full_solution: it does not hand over working code\n"
    "- consistent_with_award: it does not contradict the points awarded\n\n"
    "Write reasoning before choosing a verdict. Report confidence honestly: a "
    "borderline case should get a middling number, not a high one."
)


def _rubric_block(case: GradingCase) -> str:
    return "\n".join(
        f"- {item.id} ({item.points:g} pt): {item.description}" for item in case.rubric.items
    )


def grader_messages(case: GradingCase) -> tuple[str, str]:
    user = (
        f"Question ({case.question_id}):\n{case.question_prompt}\n\n"
        f"Rubric (total {case.rubric.max_points:g} points):\n{_rubric_block(case)}\n\n"
        f"Student response:\n{case.student_response}\n"
    )
    return GRADER_SYSTEM, user


def judge_messages(case: GradingCase, grading: ModelGrading) -> tuple[str, str]:
    required = ", ".join(case.expected.must_mention) or "(none specified)"
    forbidden = ", ".join(case.expected.must_not_mention) or "(none specified)"
    correct = ", ".join(sorted(case.expected.awarded_item_ids)) or "(none)"
    awarded = ", ".join(sorted(grading.awarded_item_ids)) or "(none)"

    user = (
        f"Question ({case.question_id}):\n{case.question_prompt}\n\n"
        f"Rubric:\n{_rubric_block(case)}\n\n"
        f"Student response:\n{case.student_response}\n\n"
        f"Correct awards: {correct} ({case.expected_points:g} pt)\n"
        f"Grader awarded: {awarded} ({grading.total_points:g} pt)\n\n"
        f"Feedback must raise: {required}\n"
        f"Feedback must avoid: {forbidden}\n\n"
        f"Grader feedback to judge:\n{grading.feedback}\n"
    )
    return JUDGE_SYSTEM, user
