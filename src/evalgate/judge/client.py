from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic
from anthropic.types import ToolChoiceAutoParam, ToolChoiceToolParam

from evalgate.core.hashing import content_hash

log = logging.getLogger(__name__)

log = logging.getLogger(__name__)

_RETRYABLE: tuple[type[Exception], ...] = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


@dataclass(frozen=True, slots=True)
class LlmRequest:
    """Everything that determines the response, and therefore the cache key.

    There is no temperature or other sampling control here because the API no
    longer accepts them. Reproducibility comes from the response cache: an
    identical request replays a stored answer rather than resampling one.
    """

    model: str
    system: str
    user: str
    tool_name: str
    tool_description: str
    input_schema: dict[str, Any]
    max_tokens: int = 1024
    # Not every model accepts tool_choice type "tool". With forcing off the
    # model may answer in prose instead, which callers have to handle.
    force_tool: bool = True

    def cache_key(self) -> str:
        return content_hash(
            {
                "model": self.model,
                "system": self.system,
                "user": self.user,
                "tool_name": self.tool_name,
                "input_schema": self.input_schema,
                "max_tokens": self.max_tokens,
                "force_tool": self.force_tool,
            }
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "system": self.system,
            "user": self.user,
            "tool_name": self.tool_name,
            "max_tokens": self.max_tokens,
            "force_tool": self.force_tool,
        }


@dataclass(frozen=True, slots=True)
class LlmResponse:
    raw_text: str
    payload: dict[str, Any] | None = None
    parse_error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    stop_reason: str | None = None


class LlmClient(Protocol):
    async def complete(self, request: LlmRequest) -> LlmResponse: ...


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429
    return False


@dataclass
class AnthropicClient:
    """Thin wrapper: tool use, bounded concurrency, retry with jitter.

    Tool use rather than "reply in JSON" because the API enforces the schema
    on its side, which removes the most common source of unparseable output.
    """

    api_key: str
    max_concurrency: int = 4
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    _client: anthropic.AsyncAnthropic = field(init=False, repr=False)
    _gate: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self._gate = asyncio.Semaphore(self.max_concurrency)

    async def aclose(self) -> None:
        await self._client.close()

    async def complete(self, request: LlmRequest) -> LlmResponse:
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                async with self._gate:
                    return await self._attempt(request)
            except Exception as exc:
                last = exc
                if not _is_retryable(exc) or attempt == self.max_attempts:
                    raise
                delay = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
                delay += random.random() * 0.25 * delay
                log.warning(
                    "llm call failed (attempt %d/%d): %r, retrying in %.1fs",
                    attempt,
                    self.max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"retries exhausted: {last!r}")

    async def _attempt(self, request: LlmRequest) -> LlmResponse:
        tool_choice: ToolChoiceToolParam | ToolChoiceAutoParam = (
            {"type": "tool", "name": request.tool_name}
            if request.force_tool
            else {"type": "auto"}
        )

        started = time.perf_counter()
        message = await self._client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            tools=[
                {
                    "name": request.tool_name,
                    "description": request.tool_description,
                    "input_schema": request.input_schema,
                }
            ],
            tool_choice=tool_choice,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        payload: dict[str, Any] | None = None
        texts: list[str] = []
        for block in message.content:
            if block.type == "tool_use" and block.name == request.tool_name:
                payload = dict(block.input) if isinstance(block.input, dict) else None
            elif block.type == "text":
                texts.append(block.text)

        raw_text = json.dumps(payload) if payload is not None else "\n".join(texts)
        parse_error = None if payload is not None else "model returned no tool_use block"

        return LlmResponse(
            raw_text=raw_text,
            payload=payload,
            parse_error=parse_error,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=latency_ms,
            stop_reason=message.stop_reason,
        )