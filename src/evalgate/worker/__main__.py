from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
from uuid import UUID

from evalgate.db.session import build_engine
from evalgate.runner.loop import worker_loop
from evalgate.runner.stub import StubGrader


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evalgate-worker")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", default=_default_worker_id())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--stale-after", type=float, default=30.0)
    parser.add_argument("--latency-ms", type=int, default=0)
    parser.add_argument("--over-credit-rate", type=float, default=0.0)
    parser.add_argument("--malformed-rate", type=float, default=0.0)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    engine = build_engine()
    grader = StubGrader(
        latency_ms=args.latency_ms,
        over_credit_rate=args.over_credit_rate,
        malformed_rate=args.malformed_rate,
    )
    try:
        stats = await worker_loop(
            engine,
            run_id=UUID(args.run_id),
            grader=grader,
            worker_id=args.worker_id,
            batch_size=args.batch_size,
            poll_interval=args.poll_interval,
            stale_after_seconds=args.stale_after,
        )
    finally:
        await engine.dispose()

    logging.info(
        "worker %s done: claimed=%d succeeded=%d failed=%d duplicates=%d reclaimed=%d",
        args.worker_id,
        stats.claimed,
        stats.succeeded,
        stats.failed,
        stats.duplicates,
        stats.reclaimed,
    )


if __name__ == "__main__":
    asyncio.run(main())