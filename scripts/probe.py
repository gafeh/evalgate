from __future__ import annotations

import argparse
from typing import Any

import anthropic

from evalgate.config import get_settings
from evalgate.judge.prompts import (
    GRADER_SCHEMA,
    GRADER_TOOL_NAME,
    JUDGE_SCHEMA,
    JUDGE_TOOL_NAME,
)

HI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

TOOLS: dict[str, tuple[str, dict[str, Any]]] = {
    "hi": ("say_hi", HI_SCHEMA),
    "grader": (GRADER_TOOL_NAME, GRADER_SCHEMA),
    "judge": (JUDGE_TOOL_NAME, JUDGE_SCHEMA),
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe")
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--tool", choices=sorted(TOOLS), default="judge")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("set ANTHROPIC_API_KEY in .env first")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    if args.list_models:
        for model in client.models.list(limit=50).data:
            print(model.id)
        return

    name, schema = TOOLS[args.tool]
    try:
        message = client.messages.create(
            model=args.model,
            max_tokens=512,
            system="You answer by calling the tool.",
            messages=[{"role": "user", "content": "Call the tool with anything valid."}],
            tools=[{"name": name, "description": "probe", "input_schema": schema}],
            tool_choice={"type": "tool", "name": name},
        )
    except Exception as exc:
        print("exception:", type(exc).__name__)
        body = getattr(getattr(exc, "response", None), "text", None)
        print("body:", body or str(exc))
        return

    print("blocks:", [block.type for block in message.content])
    for block in message.content:
        if block.type == "tool_use":
            print("input:", block.input)