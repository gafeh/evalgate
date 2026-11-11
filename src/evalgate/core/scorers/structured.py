from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any, ClassVar

import jsonschema

from evalgate.core.scorers.base import Scorer, register
from evalgate.core.types import Score


@register
@dataclass(frozen=True, slots=True)
class SetF1(Scorer[AbstractSet[str], AbstractSet[str]]):
    """F1 over set membership, with precision and recall kept separately.

    Two empty sets count as a perfect match, because "award nothing" is a
    legitimate grading decision and should not be scored as a miss.
    """

    name: ClassVar[str] = "set_f1"
    pass_threshold: float = 1.0

    def score(self, expected: AbstractSet[str], actual: AbstractSet[str]) -> Score:
        true_pos = len(expected & actual)
        false_pos = len(actual - expected)
        false_neg = len(expected - actual)

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 1.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 1.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        return Score(
            scorer=self.name,
            value=f1,
            passed=f1 >= self.pass_threshold,
            detail={
                "precision": precision,
                "recall": recall,
                "true_positives": true_pos,
                "false_positives": sorted(actual - expected),
                "false_negatives": sorted(expected - actual),
            },
        )


@register
@dataclass(frozen=True, slots=True)
class JsonSchemaValid(Scorer[Mapping[str, Any], object]):
    """expected is a JSON Schema, actual is the value (or a JSON string).

    This is the scorer that catches a model quietly drifting out of its
    output contract, which is the most common silent eval failure there is.
    """

    name: ClassVar[str] = "json_schema_valid"
    max_errors: int = 5

    def score(self, expected: Mapping[str, Any], actual: object) -> Score:
        payload = actual
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                return Score(
                    scorer=self.name,
                    value=0.0,
                    passed=False,
                    detail={"errors": [f"not valid json: {exc.msg} at pos {exc.pos}"]},
                )

        validator = jsonschema.Draft202012Validator(dict(expected))
        errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
        messages = [
            f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
            for err in errors[: self.max_errors]
        ]
        return Score(
            scorer=self.name,
            value=1.0 if not errors else 0.0,
            passed=not errors,
            detail={"error_count": len(errors), "errors": messages},
        )