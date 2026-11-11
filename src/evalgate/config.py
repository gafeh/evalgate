from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EVALGATE_",
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://evalgate:evalgate@localhost:5433/evalgate"
    test_database_url: str = (
        "postgresql+asyncpg://evalgate:evalgate@localhost:5433/evalgate_test"
    )
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()