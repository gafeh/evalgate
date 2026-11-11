from __future__ import annotations

from evalgate.core.dataset import ExpectedGrading, GradingCase, Rubric
from evalgate.db.models import Case


def hydrate_case(row: Case) -> GradingCase:
    """Rebuild the pure domain object from its stored JSONB.

    Validation runs again on the way out. If a migration or a manual edit
    ever corrupts stored JSON, this is where it surfaces.
    """
    return GradingCase(
        id=row.external_id,
        question_id=row.question_id,
        question_prompt=row.question_prompt,
        rubric=Rubric.model_validate(row.rubric),
        student_response=row.student_response,
        expected=ExpectedGrading.model_validate(row.expected),
        tags=dict(row.tags),
    )