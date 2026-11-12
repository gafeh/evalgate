from __future__ import annotations

import argparse
import asyncio
from uuid import UUID

from evalgate.labeller.session import REASON_SETS, enqueue_review, run_labelling


def cli() -> None:
    parser = argparse.ArgumentParser(prog="evalgate-label")
    sub = parser.add_subparsers(dest="command", required=True)

    enqueue_cmd = sub.add_parser("enqueue")
    enqueue_cmd.add_argument("--run-id", required=True)
    enqueue_cmd.add_argument(
        "--sample", type=int, default=50, help="target calibration sample size"
    )
    enqueue_cmd.add_argument("--seed", type=int, default=0)
    enqueue_cmd.add_argument("--no-triage", action="store_true")

    label_cmd = sub.add_parser("label")
    label_cmd.add_argument("--run-id")
    label_cmd.add_argument("--labeller", default="success")
    label_cmd.add_argument("--limit", type=int, default=50)
    label_cmd.add_argument("--reason", choices=sorted(REASON_SETS), default="calibration")

    args = parser.parse_args()
    try:
        if args.command == "enqueue":
            asyncio.run(
                enqueue_review(
                    UUID(args.run_id),
                    sample=args.sample,
                    triage=not args.no_triage,
                    seed=args.seed,
                )
            )
        else:
            asyncio.run(
                run_labelling(
                    UUID(args.run_id) if args.run_id else None,
                    labeller=args.labeller,
                    limit=args.limit,
                    reason=args.reason,
                )
            )
    except KeyboardInterrupt:
        print("\ninterrupted")


if __name__ == "__main__":
    cli()