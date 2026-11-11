from __future__ import annotations

import abc
from typing import Any, ClassVar

from evalgate.core.types import Score


class Scorer[E, A](abc.ABC):
    """A pure comparison between an expected value and an actual value.

    Scorers hold configuration only. They never read files, call APIs,
    or touch a database, which is what makes them cheap to test and safe
    to run inside a worker.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def score(self, expected: E, actual: A) -> Score:
        """Compare and return a Score."""


_REGISTRY: dict[str, type[Scorer[Any, Any]]] = {}


def register[C: type[Scorer[Any, Any]]](cls: C) -> C:
    name = cls.name
    if name in _REGISTRY:
        raise ValueError(f"duplicate scorer name: {name}")
    _REGISTRY[name] = cls
    return cls


def get_scorer_class(name: str) -> type[Scorer[Any, Any]]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown scorer {name!r}, known: {registered_scorers()}") from None


def registered_scorers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))