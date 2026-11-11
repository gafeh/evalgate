from __future__ import annotations

from evalgate.db.review import ReviewTask

RULE = "=" * 78

PROMPT = "verdict?  [p]ass  p[a]rtial  [f]ail  |  [s]kip  [q]uit  > "

ANSWERS: dict[str, str] = {
    "p": "pass",
    "pass": "pass",
    "a": "partial",
    "partial": "partial",
    "f": "fail",
    "fail": "fail",
    "s": "skip",
    "skip": "skip",
    "q": "quit",
    "quit": "quit",
    "exit": "quit",
}


def parse_answer(text: str) -> str | None:
    """Map a keypress or a whole word to an action. None means unrecognised."""
    return ANSWERS.get(text.strip().lower())


def render_task(task: ReviewTask, *, position: int, remaining: int) -> str:
    """Format one task for the terminal.

    Takes a ReviewTask, which carries no judge verdict or confidence, so
    this function cannot leak the judge's opinion into a blind label even
    by accident.
    """
    lines = [
        RULE,
        f"case {task.case_id}   ({position} done, {remaining} left)   [{task.reason}]",
        RULE,
        "",
        "QUESTION",
        task.question_prompt,
        "",
        "RUBRIC",
        *(f"  {line}" for line in task.rubric_lines),
        "",
        "STUDENT RESPONSE",
        *(f"  {line}" for line in task.student_response.splitlines() or [""]),
        "",
        "CORRECT AWARDS",
        f"  {', '.join(task.expected_awards) or '(none)'}"
        f"   = {task.expected_points:g} pt",
        "",
        "GRADER AWARDED",
        f"  {', '.join(task.grader_awards) or '(none)'}"
        f"   = {task.grader_points:g} pt",
        "",
        "GRADER FEEDBACK",
    ]

    if task.parsed_ok:
        lines.extend(f"  {line}" for line in task.grader_feedback.splitlines() or [""])
    else:
        lines.append("  (output did not parse)")
        lines.append(f"  raw: {task.raw_output[:300]}")

    lines.extend(
        [
            "",
            f"must raise: {', '.join(task.must_mention) or '(nothing specified)'}",
            f"must avoid: {', '.join(task.must_not_mention) or '(nothing specified)'}",
            "",
        ]
    )
    return "\n".join(lines)