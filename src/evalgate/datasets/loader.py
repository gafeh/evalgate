from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evalgate.core.dataset import (
    Dataset,
    ExpectedGrading,
    GradingCase,
    Rubric,
    RubricItem,
)

DEFAULT_MUST_NOT_MENTION = ("here is the full solution",)


class CorpusError(ValueError):
    """The corpus on disk does not describe a valid dataset."""


def _require(raw: dict[str, Any], key: str, path: Path) -> Any:
    if key not in raw:
        raise CorpusError(f"{path.name}: missing '{key}'")
    return raw[key]


def _normalise(text: str) -> str:
    """Collapse whitespace so cosmetic differences don't count as variety."""
    return " ".join(text.split())


def load_task_file(path: Path) -> list[GradingCase]:
    """Read one task file into grading cases.

    An authoring 'note' on a case is read and discarded. It records why a
    particular award set was chosen, which belongs in review rather than in
    a prompt sent to a model.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CorpusError(f"{path.name}: expected a mapping at the top level")

    question_id = str(_require(raw, "question_id", path))
    prompt = str(_require(raw, "prompt", path)).strip()

    rubric = Rubric(
        items=tuple(
            RubricItem(
                id=str(entry["id"]),
                description=" ".join(str(entry["description"]).split()),
                points=float(entry.get("points", 1.0)),
            )
            for entry in _require(raw, "rubric", path)
        )
    )
    task_forbidden = tuple(raw.get("must_not_mention", DEFAULT_MUST_NOT_MENTION))

    cases: list[GradingCase] = []
    for entry in _require(raw, "cases", path):
        case_id = str(_require(entry, "id", path))
        try:
            cases.append(
                GradingCase(
                    id=case_id,
                    question_id=question_id,
                    question_prompt=prompt,
                    rubric=rubric,
                    student_response=str(entry["response"]).rstrip(),
                    expected=ExpectedGrading(
                        awarded_item_ids=frozenset(entry.get("awards", [])),
                        notes=str(entry.get("note", "")),
                        must_mention=tuple(entry.get("must_mention", ())),
                        must_not_mention=tuple(
                            entry.get("must_not_mention", task_forbidden)
                        ),
                    ),
                    tags={
                        str(key): str(value)
                        for key, value in (entry.get("tags") or {}).items()
                    },
                )
            )
        except (KeyError, ValueError) as exc:
            raise CorpusError(f"{path.name}: case '{case_id}': {exc}") from exc

    if not cases:
        raise CorpusError(f"{path.name}: no cases")
    return cases


def _check_unique_ids(cases: list[GradingCase]) -> None:
    seen: set[str] = set()
    clashes: set[str] = set()
    for case in cases:
        if case.id in seen:
            clashes.add(case.id)
        seen.add(case.id)
    if clashes:
        raise CorpusError(f"duplicate case ids across the corpus: {sorted(clashes)}")


def _check_distinct_responses(cases: list[GradingCase]) -> None:
    """Reject a corpus that repeats itself.

    Duplicated cases inflate every sample size downstream: the bootstrap and
    the calibration metrics treat repeated rows as independent observations
    when they are one. Catching it here is cheaper than discovering it after
    labelling two hundred of them.
    """
    seen: dict[tuple[str, str], str] = {}
    duplicates: list[str] = []
    for case in cases:
        key = (case.question_id, _normalise(case.student_response))
        if key in seen:
            duplicates.append(f"{case.id} repeats {seen[key]}")
        else:
            seen[key] = case.id
    if duplicates:
        raise CorpusError("duplicate responses: " + "; ".join(duplicates))


def load_corpus(
    source: Path, *, name: str = "apcs-frq", version: str = "1"
) -> Dataset:
    """Load a directory of task files, or one file, into a Dataset."""
    files = sorted(source.glob("*.yaml")) if source.is_dir() else [source]
    if not files:
        raise CorpusError(f"no task files found at {source}")

    cases: list[GradingCase] = []
    for path in files:
        cases.extend(load_task_file(path))

    _check_unique_ids(cases)
    _check_distinct_responses(cases)
    return Dataset(name=name, version=version, cases=tuple(cases))