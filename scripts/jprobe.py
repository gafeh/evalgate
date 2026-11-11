from __future__ import annotations

import json
import traceback

import anthropic

from evalgate.config import get_settings
from evalgate.judge.prompts import JUDGE_SCHEMA, JUDGE_TOOL_NAME


def main() -> None:
    print("probe starting", flush=True)
    settings = get_settings()
    print("key present:", bool(settings.anthropic_api_key), flush=True)
    print("tool name:", JUDGE_TOOL_NAME, flush=True)
    print("schema:", json.dumps(JUDGE_SCHEMA), flush=True)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        message = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=1024,
            system="You answer by calling the tool.",
            messages=[{"role": "user", "content": "Call the tool with anything valid."}],
            tools=[
                {
                    "name": JUDGE_TOOL_NAME,
                    "description": "probe",
                    "input_schema": JUDGE_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": JUDGE_TOOL_NAME},
        )
    except Exception:
        traceback.print_exc()
    else:
        print("ok:", [block.type for block in message.content], flush=True)
    print("probe done", flush=True)


if __name__ == "__main__":
    main()
