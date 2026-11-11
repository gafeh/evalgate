from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_mtok_usd: float
    output_per_mtok_usd: float


# VERIFY THESE against Anthropic's current pricing page before you trust a
# cost report. A stale constant here makes every number downstream wrong,
# and nothing in the code can detect that for you.
PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(1.00, 5.00),
    "claude-sonnet-5": ModelPrice(3.00, 15.00),
    "claude-sonnet-5-5": ModelPrice(3.00, 15.00),
    "claude-opus-5": ModelPrice(15.00, 75.00),
    "claude-opus-5-5": ModelPrice(15.00, 75.00),
}

_warned: set[str] = set()


def is_priced(model: str) -> bool:
    return model in PRICES


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES.get(model)
    if price is None:
        if model not in _warned:
            _warned.add(model)
            log.warning("no price for model %s, costs will be reported as 0", model)
        return 0.0
    return (
        input_tokens * price.input_per_mtok_usd + output_tokens * price.output_per_mtok_usd
    ) / _MILLION