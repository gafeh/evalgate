import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.analysis.load import load_label_pairs
from evalgate.core.calibration import (
    LabelPair,
    agreement_rate,
    brier_score,
    compute_calibration,
    effective_sample_size,
    recommend_threshold,
)
from evalgate.core.judgment import CRITERIA, JudgeAssessment, JudgeVerdict
from evalgate.db.models import Run
from evalgate.db.repositories import (
    create_run,
    list_results,
    save_dataset,
    save_judge_assessment,
)
from evalgate.db.review import (
    SamplingConflictError,
    enqueue_calibration_sample,
    enqueue_stratified_sample,
    next_review_task,
    save_label,
)
from evalgate.db.session import session_scope
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader
from tests.factories import make_case, make_dataset

PASS = JudgeVerdict.PASS
FAIL = JudgeVerdict.FAIL


def pair(index: int, agrees: bool, confidence: float, weight: float = 1.0) -> LabelPair:
    return LabelPair(
        case_id=f"c{index}",
        judge_verdict=PASS,
        judge_confidence=confidence,
        human_verdict=PASS if agrees else FAIL,
        weight=weight,
    )


def test_equal_weights_reproduce_the_unweighted_numbers() -> None:
    """Backward compatibility: a simple random sample must not change."""
    pairs = [pair(i, i % 4 != 0, 0.9) for i in range(40)]
    weighted = [
        LabelPair(
            case_id=p.case_id,
            judge_verdict=p.judge_verdict,
            judge_confidence=p.judge_confidence,
            human_verdict=p.human_verdict,
            weight=1.0,
        )
        for p in pairs
    ]
    assert agreement_rate(pairs) == agreement_rate(weighted)
    assert brier_score(pairs) == brier_score(weighted)
    assert effective_sample_size(pairs) == 40.0


def test_weights_shift_the_estimate_toward_the_population() -> None:
    """Oversample a bad stratum and weighting must undo the oversampling."""
    # 10 high-confidence cases standing for 100, all correct.
    high = [pair(i, True, 0.95, weight=10.0) for i in range(10)]
    # 10 low-confidence cases standing for 10, all wrong.
    low = [pair(i + 10, False, 0.3, weight=1.0) for i in range(10)]

    naive = sum(1 for p in high + low if p.agrees) / 20
    assert naive == 0.5

    # The population is 100 right and 10 wrong, so about 0.909.
    assert agreement_rate(high + low) == pytest.approx(100 / 110)


def test_effective_sample_size_falls_with_unequal_weights() -> None:
    equal = [pair(i, True, 0.9, weight=1.0) for i in range(20)]
    lopsided = [pair(i, True, 0.9, weight=1.0) for i in range(10)]
    lopsided += [pair(i + 10, True, 0.9, weight=19.0) for i in range(10)]

    assert effective_sample_size(equal) == pytest.approx(20.0)
    assert effective_sample_size(lopsided) < 15.0


def test_a_threshold_cannot_rest_on_a_few_heavy_cases() -> None:
    pairs = [pair(0, True, 0.99, weight=50.0)]
    pairs += [pair(i + 1, False, 0.4, weight=1.0) for i in range(10)]
    assert recommend_threshold(pairs, target_accuracy=0.9, min_support=5) is None


def test_the_report_says_when_it_is_weighted() -> None:
    plain = compute_calibration([pair(i, True, 0.9) for i in range(40)])
    assert plain.weighted is False

    weighted = compute_calibration(
        [
            LabelPair(
                case_id=f"c{i}",
                judge_verdict=PASS,
                judge_confidence=0.9,
                human_verdict=PASS,
                weight=2.0,
                stratum="0.8-1.0",
            )
            for i in range(40)
        ]
    )
    assert weighted.weighted is True
    assert weighted.strata == {"0.8-1.0": 40}


async def _judged_run(engine: AsyncEngine, confidences: list[float]) -> Run:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session,
            make_dataset(*[make_case(f"c{i:03d}") for i in range(len(confidences))]),
        )
        run = await create_run(session, dataset=dataset, model="stub-v1")

    await worker_loop(engine, run_id=run.id, grader=StubGrader(), worker_id="w1")

    async with session_scope(engine) as session:
        for result, confidence in zip(
            await list_results(session, run.id), confidences, strict=True
        ):
            await save_judge_assessment(
                session,
                result_id=result.id,
                model="judge-test",
                assessment=JudgeAssessment(
                    verdict=PASS,
                    confidence=confidence,
                    criteria=dict.fromkeys(CRITERIA, True),
                ),
                cache_key="k" * 64,
            )
    return run


async def test_stratified_sampling_covers_a_rare_stratum(engine: AsyncEngine) -> None:
    """36 confident cases and 4 unsure ones. A random sample would miss them."""
    run = await _judged_run(engine, [0.95] * 36 + [0.3] * 4)

    async with session_scope(engine) as session:
        taken = await enqueue_stratified_sample(
            session, run_id=run.id, per_stratum=4, bins=5
        )

    assert taken == {"0.2-0.4": 4, "0.8-1.0": 4}


async def test_weights_reflect_how_hard_each_stratum_was_sampled(
    engine: AsyncEngine,
) -> None:
    run = await _judged_run(engine, [0.95] * 36 + [0.3] * 4)

    async with session_scope(engine) as session:
        await enqueue_stratified_sample(
            session, run_id=run.id, per_stratum=4, bins=5
        )
        for _ in range(8):
            task = await next_review_task(session, run_id=run.id)
            assert task is not None
            await save_label(
                session,
                review_item_id=task.review_item_id,
                labeller="tester",
                verdict=PASS,
            )
        pairs = await load_label_pairs(session, run_id=run.id)

    by_stratum = {p.stratum: p.weight for p in pairs}
    # 36 cases represented by 4, and 4 represented by 4.
    assert by_stratum["0.8-1.0"] == pytest.approx(9.0)
    assert by_stratum["0.2-0.4"] == pytest.approx(1.0)
    assert sum(p.weight for p in pairs) == pytest.approx(40.0)


async def test_mixing_designs_is_refused(engine: AsyncEngine) -> None:
    run = await _judged_run(engine, [0.95] * 10)

    async with session_scope(engine) as session:
        await enqueue_calibration_sample(session, run_id=run.id, size=4)
        with pytest.raises(SamplingConflictError, match="simple random"):
            await enqueue_stratified_sample(
                session, run_id=run.id, per_stratum=2, bins=5
            )


async def test_mixing_designs_is_refused_the_other_way(engine: AsyncEngine) -> None:
    run = await _judged_run(engine, [0.95] * 10)

    async with session_scope(engine) as session:
        await enqueue_stratified_sample(
            session, run_id=run.id, per_stratum=2, bins=5
        )
        with pytest.raises(SamplingConflictError, match="stratified"):
            await enqueue_calibration_sample(session, run_id=run.id, size=4)