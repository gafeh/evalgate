from __future__ import annotations

import hashlib
import json
from collections.abc import Set as AbstractSet
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def _fallback(value: Any) -> Any:
    if isinstance(value, AbstractSet):
        return sorted(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    raise TypeError(f"not canonicalisable: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Byte-stable JSON. Keys sorted, no padding, sets sorted.

    Sets are sorted rather than listed, because Python set iteration order
    depends on hash randomisation and would otherwise make the same dataset
    hash differently on every process start.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_fallback,
    )


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()