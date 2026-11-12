import pytest
from typer.testing import CliRunner

from evalgate.cli import app

runner = CliRunner()

# Bare groups print help and exit 2 under no_args_is_help, which is Click's
# convention for "you did not name a command".
GROUPS = [[], ["dataset"], ["run"], ["review"]]

HELP = [
    ["dataset", "load", "--help"],
    ["dataset", "list", "--help"],
    ["run", "start", "--help"],
    ["run", "list", "--help"],
    ["run", "status", "--help"],
    ["run", "compare", "--help"],
    ["review", "enqueue", "--help"],
    ["review", "label", "--help"],
    ["grade", "--help"],
    ["worker", "--help"],
    ["calibrate", "--help"],
    ["serve", "--help"],
    ["seal", "--help"],
    ["verify", "--help"],
    ["report", "--help"],
]


@pytest.mark.parametrize("argv", GROUPS)
def test_a_group_with_no_command_shows_its_help(argv: list[str]) -> None:
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "Usage:" in result.output
    assert "Commands" in result.output


@pytest.mark.parametrize("argv", HELP)
def test_every_command_is_wired(argv: list[str]) -> None:
    """Help text exercises the whole command tree without touching the database."""
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_a_bad_dataset_spec_is_rejected() -> None:
    result = runner.invoke(app, ["run", "start", "--dataset", "no-version-here"])
    assert result.exit_code != 0
    assert "name@version" in result.output


def test_an_unknown_review_reason_is_rejected() -> None:
    result = runner.invoke(app, ["review", "label", "--reason", "whatever"])
    assert result.exit_code != 0
    assert "reason must be one of" in result.output