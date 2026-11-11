from evalgate.config import Settings


def test_settings_defaults_to_local_postgres() -> None:
    settings = Settings(_env_file=None)
    assert "localhost:5433" in settings.database_url
    assert settings.test_database_url.endswith("evalgate_test")


def test_settings_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("EVALGATE_DATABASE_URL", "postgresql+asyncpg://x:y@db:5432/z")
    settings = Settings(_env_file=None)
    assert settings.database_url == "postgresql+asyncpg://x:y@db:5432/z"