from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.core.judgment import CRITERIA, JudgeAssessment, JudgeVerdict
from evalgate.db.repositories import (
    create_run,
    list_results,
    save_dataset,
    save_judge_assessment,
)
from evalgate.db.review import enqueue_calibration_sample
from evalgate.db.session import session_scope
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader
from tests.factories import make_case, make_dataset


async def _labelled_run(engine: AsyncEngine, size: int = 5) -> str:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i}") for i in range(size)])
        )
        run = await create_run(session, dataset=dataset, model="stub-v1", label="demo")
    await worker_loop(engine, run_id=run.id, grader=StubGrader(), worker_id="w1")

    async with session_scope(engine) as session:
        for result in await list_results(session, run.id):
            await save_judge_assessment(
                session,
                result_id=result.id,
                model="judge-test",
                assessment=JudgeAssessment(
                    verdict=JudgeVerdict.PASS,
                    confidence=0.9,
                    criteria=dict.fromkeys(CRITERIA, True),
                ),
                cache_key="k" * 64,
            )
        await enqueue_calibration_sample(session, run_id=run.id, size=size)
    return str(run.id)


async def test_the_root_redirects_to_the_dashboard(client: AsyncClient) -> None:
    response = await client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui"


async def test_the_runs_page_lists_runs(
    client: AsyncClient, engine: AsyncEngine
) -> None:
    run_id = await _labelled_run(engine)
    page = await client.get("/ui")
    assert page.status_code == 200
    assert "demo" in page.text
    assert run_id[:8] in page.text


async def test_the_run_page_shows_slices_and_cases(
    client: AsyncClient, engine: AsyncEngine
) -> None:
    run_id = await _labelled_run(engine)
    page = await client.get(f"/ui/runs/{run_id}")
    assert page.status_code == 200
    assert "By error mode" in page.text
    assert "off_by_one" in page.text


async def test_the_calibration_page_renders_with_nothing_labelled(
    client: AsyncClient,
) -> None:
    page = await client.get("/ui/calibration")
    assert page.status_code == 200
    assert "Nothing labelled yet" in page.text



async def test_labelling_through_the_ui_swaps_in_the_next_card(
    client: AsyncClient, engine: AsyncEngine
) -> None:
    run_id = await _labelled_run(engine, 3)
    page = await client.get("/ui/label", params={"run_id": run_id})
    assert "3 left" in page.text

    item_id = page.text.split('name="review_item_id" value="')[1].split('"')[0]
    posted = await client.post(
        "/ui/label",
        data={
            "review_item_id": item_id,
            "verdict": "pass",
            "run_id": run_id,
            "notes": "fine",
        },
    )
    assert posted.status_code == 200
    assert "2 left" in posted.text
    assert item_id not in posted.text


async def test_the_queue_empties(client: AsyncClient, engine: AsyncEngine) -> None:
    run_id = await _labelled_run(engine, 1)
    page = await client.get("/ui/label", params={"run_id": run_id})
    item_id = page.text.split('name="review_item_id" value="')[1].split('"')[0]
    posted = await client.post(
        "/ui/label", data={"review_item_id": item_id, "verdict": "pass", "run_id": run_id}
    )
    assert "Nothing left" in posted.text


async def test_a_read_only_deployment_hides_the_label_form(
    locked_client: AsyncClient, engine: AsyncEngine
) -> None:
    await _labelled_run(engine, 2)
    page = await locked_client.get("/ui/label")
    assert page.status_code == 200
    assert "read-only" in page.text
    assert "hx-post" not in page.text