from __future__ import annotations

import anthropic

from evalgate.config import get_settings
from evalgate.judge.prompts import JUDGE_SCHEMA, JUDGE_TOOL_NAME

MODELS = (
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-opus-4-8",
)


def check(client: anthropic.Anthropic, model: str, forced: bool) -> str:
    choice = (
        {"type": "tool", "name": JUDGE_TOOL_NAME} if forced else {"type": "auto"}
    )
    try:
        message = client.messages.create(
            model=model,
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
            tool_choice=choice,
        )
    except anthropic.APIStatusError as exc:
        return f"{exc.status_code}: {exc.message}"
    used = [block.type for block in message.content]
    return "tool_use" if "tool_use" in used else f"no tool call, got {used}"


def main() -> None:
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    for model in MODELS:
        for forced in (True, False):
            label = "forced" if forced else "auto  "
            print(f"{model:24} {label}  {check(client, model, forced)}", flush=True)


if __name__ == "__main__":
    main()
