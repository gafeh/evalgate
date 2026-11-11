import pytest

from evalgate.core.scorers import ExactMatch, NormalizedMatch, RegexMatch, normalize


@pytest.mark.parametrize(
    ("expected", "actual", "want"),
    [
        ("abc", "abc", True),
        ("abc", "ABC", False),
        ("abc", "abc ", False),
        ("", "", True),
    ],
)
def test_exact_match(expected: str, actual: str, want: bool) -> None:
    score = ExactMatch().score(expected, actual)
    assert score.passed is want
    assert score.value == (1.0 if want else 0.0)


@pytest.mark.parametrize(
    ("expected", "actual", "want"),
    [
        ("Returns the sum.", "returns the sum", True),
        ("off  by\tone", "off by one", True),
        ("array index", "arrayindex", False),
        ("ﬁle", "file", True),
        ("sum", "product", False),
        ("", "   ", True),
    ],
)
def test_normalized_match(expected: str, actual: str, want: bool) -> None:
    assert NormalizedMatch().score(expected, actual).passed is want


def test_normalized_match_can_keep_punctuation() -> None:
    scorer = NormalizedMatch(strip_punctuation=False)
    assert scorer.score("a, b", "a b").passed is False


def test_normalize_is_idempotent() -> None:
    once = normalize("  Hello,   WORLD!  ")
    assert normalize(once) == once


@pytest.mark.parametrize(
    ("pattern", "text", "should_match", "want"),
    [
        (r"for\s*\(", "for (int i = 0;", True, True),
        (r"while\s*\(", "for (int i = 0;", True, False),
        (r"answer key", "here is the answer key", False, False),
        (r"answer key", "no leak here", False, True),
    ],
)
def test_regex_match(pattern: str, text: str, should_match: bool, want: bool) -> None:
    scorer = RegexMatch(should_match=should_match)
    assert scorer.score(pattern, text).passed is want


def test_regex_match_ignore_case() -> None:
    assert RegexMatch(ignore_case=True).score("RETURN", "return x;").passed is True