from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from evalgate.core.scorers.base import Scorer, register
from evalgate.core.types import Score

_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(
    text: str,
    *,
    casefold: bool = True,
    strip_punctuation: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Reduce text to a comparable form. NFKC first so unicode lookalikes agree."""
    out = unicodedata.normalize("NFKC", text)
    if casefold:
        out = out.casefold()
    if strip_punctuation:
        out = out.translate(_PUNCTUATION)
    if collapse_whitespace:
        out = " ".join(out.split())
    return out.strip()


@lru_cache(maxsize=512)
def _compiled(pattern: str, ignore_case: bool) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE if ignore_case else 0)


@register
@dataclass(frozen=True, slots=True)
class ExactMatch(Scorer[str, str]):
    name: ClassVar[str] = "exact_match"

    def score(self, expected: str, actual: str) -> Score:
        hit = expected == actual
        return Score(
            scorer=self.name,
            value=1.0 if hit else 0.0,
            passed=hit,
            detail={"expected": expected, "actual": actual},
        )


@register
@dataclass(frozen=True, slots=True)
class NormalizedMatch(Scorer[str, str]):
    name: ClassVar[str] = "normalized_match"
    casefold: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True

    def score(self, expected: str, actual: str) -> Score:
        norm_expected = normalize(
            expected,
            casefold=self.casefold,
            strip_punctuation=self.strip_punctuation,
            collapse_whitespace=self.collapse_whitespace,
        )
        norm_actual = normalize(
            actual,
            casefold=self.casefold,
            strip_punctuation=self.strip_punctuation,
            collapse_whitespace=self.collapse_whitespace,
        )
        hit = norm_expected == norm_actual
        return Score(
            scorer=self.name,
            value=1.0 if hit else 0.0,
            passed=hit,
            detail={"normalized_expected": norm_expected, "normalized_actual": norm_actual},
        )


@register
@dataclass(frozen=True, slots=True)
class RegexMatch(Scorer[str, str]):
    """expected is the pattern, actual is the text searched.

    should_match=False inverts the verdict, which is how you express a
    forbidden phrase ("the model must not reveal the answer key").
    """

    name: ClassVar[str] = "regex_match"
    ignore_case: bool = False
    should_match: bool = True

    def score(self, expected: str, actual: str) -> Score:
        match = _compiled(expected, self.ignore_case).search(actual)
        found = match is not None
        hit = found is self.should_match
        return Score(
            scorer=self.name,
            value=1.0 if hit else 0.0,
            passed=hit,
            detail={
                "pattern": expected,
                "found": found,
                "should_match": self.should_match,
                "matched_text": match.group(0) if match else None,
            },
        )