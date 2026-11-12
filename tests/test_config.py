import pytest

from evalgate.config import Settings, normalise_database_url


def test_settings_defaults_to_local_postgres() -> None:
    settings = Settings(_env_file=None)
    assert "localhost:5433" in settings.database_url
    assert settings.test_database_url.endswith("evalgate_test")


def test_settings_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("EVALGATE_DATABASE_URL", "postgresql+asyncpg://x:y@db:5432/z")
    settings = Settings(_env_file=None)
    assert settings.database_url == "postgresql+asyncpg://x:y@db:5432/z"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (
            "postgres://u:p@host:5432/db",
            "postgresql+asyncpg://u:p@host:5432/db",
        ),
        (
            "postgresql://u:p@host:5432/db",
            "postgresql+asyncpg://u:p@host:5432/db",
        ),
        (
            "postgresql+asyncpg://u:p@host:5432/db",
            "postgresql+asyncpg://u:p@host:5432/db",
        ),
        (
            "postgres://u:p@host:5432/db?sslmode=disable",
            "postgresql+asyncpg://u:p@host:5432/db",
        ),
        (
            "postgres://u:p@host:5432/db?sslmode=require&application_name=x",
            "postgresql+asyncpg://u:p@host:5432/db?application_name=x",
        ),
        ("", ""),
    ],
)
def test_platform_urls_are_rewritten_for_asyncpg(given: str, expected: str) -> None:
    assert normalise_database_url(given) == expected


def test_a_platform_url_is_used_when_no_explicit_one_is_set(monkeypatch) -> None:
    monkeypatch.delenv("EVALGATE_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@db.internal:5432/app")
    settings = Settings(_env_file=None)
    assert settings.database_url == "postgresql+asyncpg://u:p@db.internal:5432/app"


def test_an_explicit_url_beats_the_platform_one(monkeypatch) -> None:
    monkeypatch.setenv("EVALGATE_DATABASE_URL", "postgresql+asyncpg://me@local/mine")
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@db.internal:5432/app")
    settings = Settings(_env_file=None)
    assert settings.database_url.endswith("/mine")


def test_read_only_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("EVALGATE_READ_ONLY", raising=False)
    assert Settings(_env_file=None).read_only is False