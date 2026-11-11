from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalgate.db.models import CallPurpose, LlmCall, LlmInvocation
from evalgate.judge.client import LlmClient, LlmRequest, LlmResponse
from evalgate.judge.pricing import cost_usd


@dataclass(frozen=True, slots=True)
class RunUsage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.calls if self.calls else 0.0


async def lookup(session: AsyncSession, cache_key: str) -> LlmInvocation | None:
    stmt = select(LlmInvocation).where(LlmInvocation.cache_key == cache_key)
    return (await session.execute(stmt)).scalar_one_or_none()


async def complete_with_cache(
    session: AsyncSession,
    client: LlmClient,
    request: LlmRequest,
    *,
    purpose: CallPurpose,
    run_id: UUID,
) -> tuple[LlmResponse, bool]:
    """Return a response plus whether it came from the cache.

    The ledger entry is written either way. A cache hit records zero cost and
    zero tokens, because nothing was spent, while still proving the call
    happened as part of this run.
    """
    key = request.cache_key()
    hit = await lookup(session, key)

    if hit is not None:
        response = LlmResponse(
            raw_text=hit.raw_text,
            payload=hit.response_payload,
            parse_error=hit.parse_error,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            stop_reason=hit.stop_reason,
        )
        await _record_call(
            session,
            run_id=run_id,
            purpose=purpose,
            model=request.model,
            cache_key=key,
            cached=True,
            response=response,
            cost=0.0,
        )
        return response, True

    response = await client.complete(request)
    cost = cost_usd(request.model, response.input_tokens, response.output_tokens)

    session.add(
        LlmInvocation(
            cache_key=key,
            purpose=purpose,
            model=request.model,
            request=request.as_json(),
            response_payload=response.payload,
            raw_text=response.raw_text,
            parse_error=response.parse_error,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost,
            latency_ms=response.latency_ms,
            stop_reason=response.stop_reason,
        )
    )
    await _record_call(
        session,
        run_id=run_id,
        purpose=purpose,
        model=request.model,
        cache_key=key,
        cached=False,
        response=response,
        cost=cost,
    )
    return response, False


async def _record_call(
    session: AsyncSession,
    *,
    run_id: UUID,
    purpose: CallPurpose,
    model: str,
    cache_key: str,
    cached: bool,
    response: LlmResponse,
    cost: float,
) -> None:
    session.add(
        LlmCall(
            run_id=run_id,
            purpose=purpose,
            model=model,
            cache_key=cache_key,
            cached=cached,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost,
            latency_ms=response.latency_ms,
        )
    )
    await session.commit()


async def run_usage(session: AsyncSession, run_id: UUID) -> RunUsage:
    stmt = select(
        LlmCall.cached,
        func.count(),
        func.coalesce(func.sum(LlmCall.input_tokens), 0),
        func.coalesce(func.sum(LlmCall.output_tokens), 0),
        func.coalesce(func.sum(LlmCall.cost_usd), 0.0),
    ).where(LlmCall.run_id == run_id).group_by(LlmCall.cached)

    calls = 0
    hits = 0
    input_tokens = 0
    output_tokens = 0
    cost = 0.0
    for cached, count, in_tok, out_tok, spend in (await session.execute(stmt)).all():
        calls += count
        if cached:
            hits += count
        input_tokens += int(in_tok)
        output_tokens += int(out_tok)
        cost += float(spend)

    return RunUsage(
        calls=calls,
        cache_hits=hits,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )