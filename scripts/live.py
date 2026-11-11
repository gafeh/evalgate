from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from evalgate.config import get_settings
from evalgate.core.dataset import Dataset
from evalgate.datasets.loader import load_corpus
from evalgate.datasets.synthetic import synthetic_dataset
from evalgate.db.repositories import create_run, list_results, save_dataset
from evalgate.db.session import build_engine, session_scope
from evalgate.judge.assessor import LlmJudge
from evalgate.judge.cache import run_usage
from evalgate.judge.client import AnthropicClient
from evalgate.runner.llm_grader import LlmGrader
from evalgate.runner.loop import worker_loop


def build_dataset(corpus: str, size: int, version: str) -> Dataset:
    """Load the authored corpus, or fall back to the synthetic generator.

    The synthetic generator cycles a handful of templates, so it is for
    exercising the queue rather than for measuring anything. Only --size
    applies to it.
    """
    if corpus == "synthetic":
        return synthetic_dataset(size, version)
    return load_corpus(Path(corpus), name="apcs-frq", version=version)


async def main(
    corpus: str,
    size: int,
    version: str,
    grader_model: str,
    judge_model: str,
    no_judge: bool,
    judge_auto_tool: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("set ANTHROPIC_API_KEY in .env first")

    source = build_dataset(corpus, size, version)
    engine = build_engine()
    client = AnthropicClient(api_key=settings.anthropic_api_key, max_concurrency=3)

    async with session_scope(engine) as session:
        dataset = await save_dataset(session, source)
        run = await create_run(
            session, dataset=dataset, model=grader_model, label="live"
        )
    print(f"dataset {dataset.name}@{dataset.version} cases={dataset.case_count}")
    print(f"run id: {run.id}")

    grader = LlmGrader(client=client, engine=engine, run_id=run.id, model=grader_model)
    judge = (
        None
        if no_judge
        else LlmJudge(
            client=client,
            engine=engine,
            run_id=run.id,
            model=judge_model,
            force_tool=not judge_auto_tool,
        )
    )

    try:
        stats = await worker_loop(
            engine,
            run_id=run.id,
            grader=grader,
            worker_id="live",
            batch_size=3,
            judge=judge,
        )
    finally:
        await client.aclose()

    async with session_scope(engine) as session:
        results = await list_results(session, run.id)
        usage = await run_usage(session, run.id)
        scored = [r for r in results if r.overall is not None]
        mean = sum(r.overall or 0.0 for r in scored) / len(scored) if scored else 0.0
        over = sum(r.over_credit_points or 0.0 for r in scored)
        under = sum(r.under_credit_points or 0.0 for r in scored)

    await engine.dispose()

    print(
        f"graded {stats.succeeded}, judged {stats.judged}, "
        f"judge failures {stats.judge_failures}"
    )
    print(f"unparseable outputs: {len(results) - len(scored)}")
    print(f"mean overall: {mean:.4f}")
    print(f"over-credited points: {over}")
    print(f"under-credited points: {under}")
    print(
        f"calls {usage.calls} (cache hits {usage.cache_hits}), "
        f"tokens {usage.input_tokens} in / {usage.output_tokens} out, "
        f"cost ${usage.cost_usd:.4f}"
    )
    for problem in stats.errors[:5]:
        print(f"error: {problem}")


def cli() -> None:
    parser = argparse.ArgumentParser(prog="live")
    parser.add_argument(
        "--corpus",
        default="datasets/apcs",
        help="path to a corpus directory or file, or 'synthetic'",
    )
    parser.add_argument(
        "--size", type=int, default=4, help="only used when --corpus synthetic"
    )
    parser.add_argument("--version", default="live-1")
    parser.add_argument("--grader-model", default="claude-haiku-5-5")
    parser.add_argument("--judge-model", default="claude-sonnet-5")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--judge-auto-tool",
        action="store_true",
        help="use tool_choice auto, needed for models that reject forced tool use",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            args.corpus,
            args.size,
            args.version,
            args.grader_model,
            args.judge_model,
            args.no_judge,
            args.judge_auto_tool,
        )
    )


if __name__ == "__main__":
    cli()