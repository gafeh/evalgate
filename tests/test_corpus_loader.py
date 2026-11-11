import re
from pathlib import Path

import pytest

from evalgate.datasets.loader import CorpusError, load_corpus, load_task_file

CORPUS = Path(__file__).resolve().parents[1] / "datasets" / "apcs"


def test_the_real_corpus_loads() -> None:
    dataset = load_corpus(CORPUS)
    assert len(dataset.cases) >= 15
    assert dataset.content_hash()


def test_every_award_exists_in_its_rubric() -> None:
    """GradingCase validates this, so a bad award set fails at load time."""
    dataset = load_corpus(CORPUS)
    for case in dataset.cases:
        assert case.expected.awarded_item_ids <= case.rubric.item_ids


def test_the_corpus_covers_a_spread_of_scores() -> None:
    dataset = load_corpus(CORPUS)
    points = {case.expected_points for case in dataset.cases}
    assert len(points) >= 4, f"only {points} distinct point totals"


def test_the_corpus_covers_several_error_modes() -> None:
    dataset = load_corpus(CORPUS)
    assert len(dataset.slice_values("error_mode")) >= 5


def test_authoring_notes_do_not_reach_the_prompt() -> None:
    dataset = load_corpus(CORPUS)
    for case in dataset.cases:
        assert "note" not in case.question_prompt
        assert case.expected.notes != case.student_response


def test_duplicate_responses_are_rejected(tmp_path: Path) -> None:
    task = tmp_path / "dupes.yaml"
    task.write_text(
        "question_id: q\n"
        "prompt: do a thing\n"
        "rubric:\n"
        "  - {id: r1, points: 1, description: does the thing}\n"
        "cases:\n"
        "  - {id: a, awards: [r1], response: 'return 1;'}\n"
        "  - {id: b, awards: [r1], response: 'return   1;'}\n",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="duplicate responses"):
        load_corpus(task)


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    task = tmp_path / "ids.yaml"
    task.write_text(
        "question_id: q\n"
        "prompt: do a thing\n"
        "rubric:\n"
        "  - {id: r1, points: 1, description: does the thing}\n"
        "cases:\n"
        "  - {id: a, awards: [r1], response: 'return 1;'}\n"
        "  - {id: a, awards: [r1], response: 'return 2;'}\n",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="duplicate case ids"):
        load_corpus(task)


def test_an_award_outside_the_rubric_is_rejected(tmp_path: Path) -> None:
    task = tmp_path / "bad.yaml"
    task.write_text(
        "question_id: q\n"
        "prompt: do a thing\n"
        "rubric:\n"
        "  - {id: r1, points: 1, description: does the thing}\n"
        "cases:\n"
        "  - {id: a, awards: [r9], response: 'return 1;'}\n",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="unknown rubric items"):
        load_corpus(task)


def test_a_missing_key_names_the_file(tmp_path: Path) -> None:
    task = tmp_path / "thin.yaml"
    task.write_text("question_id: q\nprompt: hi\n", encoding="utf-8")
    with pytest.raises(CorpusError, match=re.escape("thin.yaml: missing 'rubric'")):
        load_task_file(task)