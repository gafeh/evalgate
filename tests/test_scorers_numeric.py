import pytest

from evalgate.core.scorers import NumericTolerance


@pytest.mark.parametrize(
    ("expected", "actual", "abs_tol", "want"),
    [
        (3.0, 3.0, 0.0, True),
        (3.0, 3.5, 0.0, False),
        (3.0, 3.5, 0.5, True),
        (3.0, 2.5, 0.5, True),
        (3.0, 3.6, 0.5, False),
        (0.0, 0.0, 0.0, True),
    ],
)
def test_numeric_tolerance(expected: float, actual: float, abs_tol: float, want: bool) -> None:
    assert NumericTolerance(abs_tol=abs_tol).score(expected, actual).passed is want


def test_signed_error_direction_survives() -> None:
    over = NumericTolerance().score(3.0, 5.0)
    under = NumericTolerance().score(3.0, 1.0)
    assert over.detail["signed_error"] > 0
    assert under.detail["signed_error"] < 0
    assert over.detail["abs_error"] == under.detail["abs_error"]