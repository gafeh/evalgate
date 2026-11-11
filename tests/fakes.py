from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from evalgate.judge.client import LlmRequest, LlmResponse


@dataclass
class FakeLlmClient:
    """Scripted client. Records every request so tests can assert call counts."""

    payload: dict[str, Any] | None = None
    payload_by_tool: dict[str, dict[str, Any]] = field(default_factory=dict)
    input_tokens: int = 100
    output_tokens: int = 50
    raise_on_call: Exception | None = None
    requests: list[LlmRequest] = field(default_factory=list)

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if self.raise_on_call is not None:
            raise self.raise_on_call

        payload = self.payload_by_tool.get(request.tool_name, self.payload)
        if payload is None:
            return LlmResponse(
                raw_text="I would rather not use a tool.",
                parse_error="model returned no tool_use block",
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                latency_ms=7,
            )

        return LlmResponse(
            raw_text=json.dumps(payload),
            payload=payload,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=7,
        )
