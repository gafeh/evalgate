from dataclasses import fields

from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.analysis.load import load_label_pairs
from evalgate.core.judgment import CRITERIA, JudgeAssessment, JudgeVerdict
from evalgate.db.models import Run
from evalgate.db.repositories import (
    create_run,
    list_results,
    save_dataset,
    save_judge_assessment,
)
from evalgate.db.review import (
    ReviewTask,
    enqueue_calibration_sample,
    enqueue_triage,
    next_review_task,
    review_counts,
    save_label,
    skip_review_item,
)
from evalgate.db.session import session_scope
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader
from tests.factories import make_case, make_dataset


async def _judged_run(
    engine: AsyncEngine,
    size: int,
    *,
    over_credit_rate: float = 0.0,
    malformed_rate: float = 0.0,
    confidence: float = 0.9,
    verdict: JudgeVerdict = JudgeVerdict.PASS,
) -> Run:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i:03d}") for i in range(size)])
        )
        run = await create_run(session, dataset=dataset, model="stub-v1")

    await worker_loop(
        engine,
        run_id=run.id,
        grader=StubGrader(
            over_credit_rate=over_credit_rate, malformed_rate=malformed_rate
        ),
        worker_id="w1",
    )

    async with session_scope(engine) as session:
        for result in await list_results(session, run.id):
            await save_judge_assessment(
                session,
                result_id=result.id,
                model="judge-test",
                assessment=JudgeAssessment(
                    verdict=verdict,
                    confidence=confidence,
                    criteria=dict.fromkeys(CRITERIA, True),
                ),
                cache_key="k" * 64,
            )
    return run


def test_review_task_cannot_leak_the_judges_opinion() -> None:
    """The blind-labelling guarantee, enforced by the type rather than the UI."""
    names = {field.name for field in fields(ReviewTask)}
    assert not {name for name in names if "judge" in name or "confidence" in name}
    assert "grader_feedback" in names
    assert "expected_awards" in names


async def test_calibration_sample_is_bounded_and_deterministic(
    engine: AsyncEngine,
) -> None:
    run = await _judged_run(engine, 20)
    async with session_scope(engine) as session:
        taken = await enqueue_calibration_sample(
            session, run_id=run.id, size=8, seed=1
        )
        counts = await review_counts(session, run_id=run.id)

    assert taken == 8
    assert counts == {"pending/calibration": 8}


async def test_calibration_sample_cannot_exceed_whats_available(
    engine: AsyncEngine,
) -> None:
    run = await _judged_run(engine, 3)
    async with session_scope(engine) as session:
        assert await enqueue_calibration_sample(session, run_id=run.id, size=50) == 3


async def test_a_result_is_queued_at_most_once(engine: AsyncEngine) -> None:
    run = await _judged_run(engine, 10, confidence=0.1)
    async with session_scope(engine) as session:
        sampled = await enqueue_calibration_sample(session, run_id=run.id, size=10)
        triaged = await enqueue_triage(session, run_id=run.id)

    assert sampled == 10
    assert triaged == 0


async def test_triage_picks_up_over_crediting_as_a_safety_case(
    engine: AsyncEngine,
) -> None:
    run = await _judged_run(engine, 6, over_credit_rate=1.0)
    async with session_scope(engine) as session:
        queued = await enqueue_triage(session, run_id=run.id)
        counts = await review_counts(session, run_id=run.id)

    assert queued == 6
    assert counts == {"pending/safety": 6}


async def test_triage_picks_up_unparseable_output(engine: AsyncEngine) -> None:
    run = await _judged_run(engine, 5, malformed_rate=1.0)
    async with session_scope(engine) as session:
        await enqueue_triage(session, run_id=run.id)
        counts = await review_counts(session, run_id=run.id)

    assert counts == {"pending/unparseable": 5}


async def test_confident_clean_results_are_not_triaged(engine: AsyncEngine) -> None:
    run = await _judged_run(engine, 5, confidence=0.99)
    async with session_scope(engine) as session:
        assert await enqueue_triage(session, run_id=run.id) == 0


async def test_low_confidence_results_are_triaged(engine: AsyncEngine) -> None:
    run = await _judged_run(engine, 5, confidence=0.4)
    async with session_scope(engine) as session:
        await enqueue_triage(session, run_id=run.id)
        counts = await review_counts(session, run_id=run.id)

    assert counts == {"pending/low_confidence": 5}


async def test_the_queue_serves_the_most_alarming_item_first(
    engine: AsyncEngine,
) -> None:
    low = await _judged_run(engine, 2, confidence=0.3)
    unparseable = await _judged_run(engine, 2, malformed_rate=1.0)

    async with session_scope(engine) as session:
        await enqueue_triage(session, run_id=low.id)
        await enqueue_triage(session, run_id=unparseable.id)
        task = await next_review_task(session)

    assert task is not None
    assert task.parsed_ok is False


async def test_labelling_removes_an_item_from_the_queue(engine: AsyncEngine) -> None:
    run = await _judged_run(engine, 3)
    async with session_scope(engine) as session:
        await enqueue_calibration_sample(session, run_id=run.id, size=3)
        first = await next_review_task(session, run_id=run.id)
        assert first is not None
        await save_label(
            session,
            review_item_id=first.review_item_id,
            labeller="tester",
            verdict=JudgeVerdict.PASS,
            notes="fine",
            seconds_spent=12,
        )
        second = await next_review_task(session, run_id=run.id)
        counts = await review_counts(session, run_id=run.id)

    assert second is not None
    assert second.review_item_id != first.review_item_id
    assert counts == {"labelled/calibration": 1, "pending/calibration": 2}


async def test_skipping_also_clears_an_item(engine: AsyncEngine) -> None:
    run = await _judged_run(engine, 2)
    async with session_scope(engine) as session:
        await enqueue_calibration_sample(session, run_id=run.id, size=2)
        task = await next_review_task(session, run_id=run.id)
        assert task is not None
        await skip_review_item(session, task.review_item_id)
        counts = await review_counts(session, run_id=run.id)

    assert counts == {"pending/calibration": 1, "skipped/calibration": 1}


async def test_an_empty_queue_returns_nothing(engine: AsyncEngine) -> None:
    async with session_scope(engine) as session:
        assert await next_review_task(session) is None


async def test_label_pairs_exclude_triaged_items_by_default(
    engine: AsyncEngine,
) -> None:
    """Triaged items are a biased sample and must not reach the calibration set."""
    run = await _judged_run(engine, 10, confidence=0.3)

    async with session_scope(engine) as session:
        await enqueue_calibration_sample(session, run_id=run.id, size=4, seed=2)
        await enqueue_triage(session, run_id=run.id)

        for _ in range(10):
            task = await next_review_task(session, run_id=run.id)
            if task is None:
                break
            await save_label(
                session,
                review_item_id=task.review_item_id,
                labeller="tester",
                verdict=JudgeVerdict.FAIL,
                seconds_spent=1,
            )

        calibration_only = await load_label_pairs(session, run_id=run.id)
        everything = await load_label_pairs(
            session, run_id=run.id, calibration_only=False
        )

    assert len(calibration_only) == 4
    assert len(everything) == 10
    assert all(pair.judge_confidence == 0.3 for pair in calibration_only)
async def test_enqueue_is_idempotent_because_sample_is_a_target(
    engine: AsyncEngine,
) -> None:
    run = await _judged_run(engine, 20)
    async with session_scope(engine) as session:
        first = await enqueue_calibration_sample(session, run_id=run.id, size=8)
        second = await enqueue_calibration_sample(session, run_id=run.id, size=8)
        topped_up = await enqueue_calibration_sample(session, run_id=run.id, size=12)
        counts = await review_counts(session, run_id=run.id)

    assert first == 8
    assert second == 0
    assert topped_up == 4
    assert counts == {"pending/calibration": 12}