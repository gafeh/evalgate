from sqlalchemy.ext.asyncio import AsyncSession

from evalgate.core.grading import ModelGrading, evaluate_grading
from evalgate.db.repositories import (
    create_run,
    list_cases,
    list_results,
    record_result,
    save_dataset,
)
from tests.factories import make_case, make_dataset


async def test_record_result_persists_scores_and_credit_split(session: AsyncSession) -> None:
    core_case = make_case("a")
    dataset = await save_dataset(session, make_dataset(core_case))
    run = await create_run(session, dataset=dataset, model="claude-test")
    case_row = (await list_cases(session, dataset.id))[0]

    grading = ModelGrading(
        awarded_item_ids=frozenset({"r1", "r2", "r3"}),
        total_points=4.0,
        feedback="Looks perfect!",
    )
    evaluation = evaluate_grading(core_case, grading)

    result = await record_result(
        session,
        run_id=run.id,
        case_id=case_row.id,
        raw_output=grading.model_dump_json(),
        evaluation=evaluation,
        grading=grading.model_dump(mode="json"),
        latency_ms=1234,
    )

    assert result.parsed_ok is True
    assert result.over_credit_points == 2.0
    assert result.under_credit_points == 0.0
    assert result.scores["set_f1"]["value"] == 0.8
    assert result.scores["numeric_tolerance"]["passed"] is False
    assert len(result.content_hash) == 64


async def test_unparseable_output_still_records_a_result(session: AsyncSession) -> None:
    dataset = await save_dataset(session, make_dataset(make_case("a")))
    run = await create_run(session, dataset=dataset, model="claude-test")
    case_row = (await list_cases(session, dataset.id))[0]

    result = await record_result(
        session,
        run_id=run.id,
        case_id=case_row.id,
        raw_output="Sure! Here is the grading:",
        parse_error="no json object found in response",
    )

    assert result.parsed_ok is False
    assert result.overall is None
    assert result.scores == {}


async def test_list_results_scoped_to_its_run(session: AsyncSession) -> None:
    dataset = await save_dataset(session, make_dataset(make_case("a")))
    run_one = await create_run(session, dataset=dataset, model="m1")
    run_two = await create_run(session, dataset=dataset, model="m2")
    case_row = (await list_cases(session, dataset.id))[0]

    await record_result(
        session, run_id=run_one.id, case_id=case_row.id, raw_output="{}", parse_error="x"
    )

    assert len(await list_results(session, run_one.id)) == 1
    assert len(await list_results(session, run_two.id)) == 0