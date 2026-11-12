from collections.abc import AsyncIterator
from dataclasses import fields

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate.api.app import create_app
from evalgate.api.schemas import ReviewTaskOut
from evalgate.core.judgment import CRITERIA, JudgeAssessment, JudgeVerdict
from evalgate.db.repositories import (
    create_run,
    list_results,
    save_dataset,
    save_judge_assessment,
)
from evalgate.db.review import ReviewTask
from evalgate.db.session import session_scope
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader
from tests.factories import make_case, make_dataset


@pytest_asyncio.fixture
async def client(engine: AsyncEngine) -> AsyncIterator[AsyncClient]:
    app = create_app(engine=engine)
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as http,
        app.router.lifespan_context(app),
    ):
        yield http


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_dataset_round_trip(client: AsyncClient) -> None:
    payload = make_dataset(make_case("a"), make_case("b")).model_dump(mode="json")
    created = await client.post("/datasets", json=payload)
    assert created.status_code == 201
    assert created.json()["case_count"] == 2

    again = await client.post("/datasets", json=payload)
    assert again.status_code == 201

    listed = await client.get("/datasets")
    assert len(listed.json()) == 1

    shown = await client.get("/datasets/apcs-frq-smoke/1")
    assert shown.json()["content_hash"] == created.json()["content_hash"]


async def test_changed_content_under_the_same_version_conflicts(
    client: AsyncClient,
) -> None:
    await client.post("/datasets", json=make_dataset(make_case("a")).model_dump(mode="json"))
    changed = make_dataset(make_case("a"), make_case("b")).model_dump(mode="json")
    response = await client.post("/datasets", json=changed)
    assert response.status_code == 409
    assert "Bump the version" in response.json()["detail"]


async def test_missing_dataset_is_404(client: AsyncClient) -> None:
    assert (await client.get("/datasets/nope/1")).status_code == 404


async def test_creating_a_run_enqueues_without_grading(client: AsyncClient) -> None:
    await client.post(
        "/datasets", json=make_dataset(make_case("a"), make_case("b")).model_dump(mode="json")
    )
    created = await client.post(
        "/runs",
        json={
            "dataset_name": "apcs-frq-smoke",
            "dataset_version": "1",
            "model": "stub-v1",
            "label": "baseline",
        },
    )
    assert created.status_code == 202
    run_id = created.json()["id"]

    detail = await client.get(f"/runs/{run_id}")
    body = detail.json()
    assert body["work_items"] == {"queued": 2}
    assert body["summary"]["cases"] == 0
    assert body["usage"]["calls"] == 0


async def test_a_run_against_a_missing_dataset_is_404(client: AsyncClient) -> None:
    response = await client.post(
        "/runs",
        json={"dataset_name": "ghost", "dataset_version": "1", "model": "stub-v1"},
    )
    assert response.status_code == 404


async def test_run_detail_reports_results_and_slices(
    client: AsyncClient, engine: AsyncEngine
) -> None:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i}") for i in range(6)])
        )
        run = await create_run(session, dataset=dataset, model="stub-v1")
    await worker_loop(
        engine, run_id=run.id, grader=StubGrader(over_credit_rate=0.5), worker_id="w1"
    )

    body = (await client.get(f"/runs/{run.id}")).json()
    assert body["work_items"] == {"done": 6}
    assert body["summary"]["cases"] == 6
    assert body["summary"]["over_credit_points"] > 0
    assert body["slices"][0]["tag_value"] == "off_by_one"

    results = (await client.get(f"/runs/{run.id}/results")).json()
    assert len(results) == 6
    assert results[0]["judge_verdict"] is None


async def test_compare_gates_two_runs(client: AsyncClient, engine: AsyncEngine) -> None:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i:03d}") for i in range(40)])
        )
        baseline = await create_run(session, dataset=dataset, model="stub-v1")
        candidate = await create_run(session, dataset=dataset, model="stub-v2")

    await worker_loop(engine, run_id=baseline.id, grader=StubGrader(), worker_id="w1")
    await worker_loop(
        engine,
        run_id=candidate.id,
        grader=StubGrader(model_name="stub-v2", over_credit_rate=1.0),
        worker_id="w2",
    )

    body = (await client.get(f"/runs/{candidate.id}/compare/{baseline.id}")).json()
    assert body["paired_cases"] == 40
    assert body["verdict"] == "regressed"
    assert body["passed"] is False
    assert body["over_credit_delta"]["low"] > 0


async def test_review_flow_over_http(client: AsyncClient, engine: AsyncEngine) -> None:
    async with session_scope(engine) as session:
        dataset = await save_dataset(
            session, make_dataset(*[make_case(f"c{i}") for i in range(5)])
        )
        run = await create_run(session, dataset=dataset, model="stub-v1")
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

    enqueued = await client.post(f"/runs/{run.id}/review", json={"sample": 3})
    assert enqueued.json()["added_to_sample"] == 3

    task = (await client.get("/review/next", params={"run_id": str(run.id)})).json()
    assert "judge_verdict" not in task
    assert "judge_confidence" not in task

    labelled = await client.post(
        f"/review/{task['review_item_id']}/label",
        json={"labeller": "tester", "verdict": "pass", "notes": "fine"},
    )
    assert labelled.status_code == 204

    counts = (await client.get("/review/counts", params={"run_id": str(run.id)})).json()
    assert counts["labelled/calibration"] == 1

    report = (
        await client.get("/reports/calibration", params={"run_id": str(run.id)})
    ).json()
    assert report["pairs"] == 1
    assert report["trustworthy"] is False
    assert report["warnings"]


async def test_the_review_endpoint_cannot_leak_the_judge() -> None:
    """Same guarantee as ReviewTask, enforced on the wire."""
    names = set(ReviewTaskOut.model_fields)
    assert not {name for name in names if "judge" in name or "confidence" in name}
    assert "grader_feedback" in names
    assert {field.name for field in fields(ReviewTask)} - {"raw_output"} <= names


async def test_an_unknown_verdict_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/review/00000000-0000-0000-0000-000000000000/label",
        json={"labeller": "tester", "verdict": "maybe"},
    )
    assert response.status_code == 422


async def test_an_empty_review_queue_is_404(client: AsyncClient) -> None:
    assert (await client.get("/review/next")).status_code == 404
@pytest_asyncio.fixture
async def locked_client(engine: AsyncEngine) -> AsyncIterator[AsyncClient]:
    app = create_app(engine=engine, read_only=True)
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as http,
        app.router.lifespan_context(app),
    ):
        yield http


async def test_a_read_only_deployment_still_serves_reads(
    locked_client: AsyncClient,
) -> None:
    health = await locked_client.get("/health")
    assert health.status_code == 200
    assert health.json()["read_only"] is True
    assert (await locked_client.get("/runs")).status_code == 200


async def test_a_read_only_deployment_rejects_every_write(
    locked_client: AsyncClient,
) -> None:
    """A public demo must not let a stranger start a run or write a label."""
    writes = [
        locked_client.post("/datasets", json={}),
        locked_client.post("/runs", json={}),
        locked_client.post(
            "/runs/00000000-0000-0000-0000-000000000000/review", json={}
        ),
        locked_client.post(
            "/review/00000000-0000-0000-0000-000000000000/label", json={}
        ),
    ]
    for response in writes:
        result = await response
        assert result.status_code == 405
        assert result.json()["detail"] == "this deployment is read-only"


async def test_writes_still_work_when_not_locked(client: AsyncClient) -> None:
    payload = make_dataset(make_case("a")).model_dump(mode="json")
    assert (await client.post("/datasets", json=payload)).status_code == 201