from __future__ import annotations

import argparse
import asyncio
from uuid import UUID

from evalgate.datasets.synthetic import synthetic_dataset
from evalgate.db.queue import outstanding_count
from evalgate.db.repositories import (
    create_run,
    list_results,
    save_dataset,
    work_item_counts,
)
from evalgate.db.session import build_engine, session_scope


async def seed(size: int, version: str) -> None:
    engine = build_engine()
    async with session_scope(engine) as session:
        dataset = await save_dataset(session, synthetic_dataset(size, version))
        run = await create_run(session, dataset=dataset, model="stub-v1", label="demo")
        print(f"dataset {dataset.name}@{dataset.version} cases={dataset.case_count}")
        print(f"run id: {run.id}")
    await engine.dispose()


async def status(run_id: UUID) -> None:
    engine = build_engine()
    async with session_scope(engine) as session:
        counts = await work_item_counts(session, run_id)
        results = await list_results(session, run_id)
        scored = [r for r in results if r.overall is not None]
        print(f"work items: {counts}")
        print(f"outstanding: {await outstanding_count(session, run_id)}")
        print(f"results: {len(results)} ({len(results) - len(scored)} unparseable)")
        if scored:
            mean = sum(r.overall or 0.0 for r in scored) / len(scored)
            over = sum(r.over_credit_points or 0.0 for r in scored)
            print(f"mean overall: {mean:.4f}")
            print(f"total over-credited points: {over}")
    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="demo")
    sub = parser.add_subparsers(dest="command", required=True)

    seed_cmd = sub.add_parser("seed")
    seed_cmd.add_argument("--size", type=int, default=500)
    seed_cmd.add_argument("--version", default="1")

    status_cmd = sub.add_parser("status")
    status_cmd.add_argument("run_id")

    args = parser.parse_args()
    if args.command == "seed":
        asyncio.run(seed(args.size, args.version))
    else:
        asyncio.run(status(UUID(args.run_id)))


if __name__ == "__main__":
    main()