from __future__ import annotations

import time
from uuid import UUID

from evalgate.core.judgment import JudgeVerdict
from evalgate.db.models import ReviewReason, ReviewStatus
from evalgate.db.review import (
    TRIAGE_REASONS,
    count_items,
    enqueue_calibration_sample,
    enqueue_triage,
    next_review_task,
    review_counts,
    save_label,
    skip_review_item,
)
from evalgate.db.session import build_engine, session_scope
from evalgate.labeller.render import PROMPT, parse_answer, render_task

REASON_SETS: dict[str, tuple[ReviewReason, ...] | None] = {
    "calibration": (ReviewReason.CALIBRATION,),
    "triage": TRIAGE_REASONS,
    "all": None,
}


async def enqueue_review(
    run_id: UUID, *, sample: int, triage: bool = True, seed: int = 0
) -> None:
    engine = build_engine()
    async with session_scope(engine) as session:
        added = await enqueue_calibration_sample(
            session, run_id=run_id, size=sample, seed=seed
        )
        total = await count_items(
            session, run_id=run_id, reasons=(ReviewReason.CALIBRATION,)
        )
        triaged = await enqueue_triage(session, run_id=run_id) if triage else 0
        counts = await review_counts(session, run_id=run_id)
    await engine.dispose()

    print(f"calibration sample: {total} (added {added} this call)")
    print(f"triaged: {triaged} new")
    print(f"queue: {counts}")


def _ask(prompt: str) -> str | None:
    """Read one line. None means the user wants out."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _ask_verdict() -> str:
    """Re-prompt until the answer parses. Returns 'quit' if the user bails."""
    while True:
        raw = _ask(PROMPT)
        if raw is None:
            return "quit"
        action = parse_answer(raw)
        if action is not None:
            return action
        print(f"'{raw.strip()}' is not a verdict. p, a, f, s or q.")


async def run_labelling(
    run_id: UUID | None, *, labeller: str, limit: int, reason: str = "calibration"
) -> None:
    reasons = REASON_SETS[reason]
    engine = build_engine()
    done = 0

    try:
        while done < limit:
            async with session_scope(engine) as session:
                task = await next_review_task(session, run_id=run_id, reasons=reasons)
                if task is None:
                    print(f"nothing left to label in the {reason} queue")
                    break
                pending = await count_items(
                    session,
                    run_id=run_id,
                    reasons=reasons,
                    status=ReviewStatus.PENDING,
                )

            print(render_task(task, position=done, remaining=pending))

            started = time.monotonic()
            action = _ask_verdict()
            spent = int(time.monotonic() - started)

            if action == "quit":
                break
            if action == "skip":
                async with session_scope(engine) as session:
                    await skip_review_item(session, task.review_item_id)
                continue

            notes = _ask("notes (enter to skip) > ")
            if notes is None:
                print("stopping without saving this one")
                break

            async with session_scope(engine) as session:
                await save_label(
                    session,
                    review_item_id=task.review_item_id,
                    labeller=labeller,
                    verdict=JudgeVerdict(action),
                    notes=notes.strip(),
                    seconds_spent=spent,
                )
            done += 1
    finally:
        await engine.dispose()

    print(f"labelled {done} this sitting")