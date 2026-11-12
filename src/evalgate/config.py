from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ASYNC_SCHEME = "postgresql+asyncpg"
LOCAL_URL = "postgresql+asyncpg://evalgate:evalgate@localhost:5433/evalgate"
LOCAL_TEST_URL = "postgresql+asyncpg://evalgate:evalgate@localhost:5433/evalgate_test"

# asyncpg takes these as connect() arguments rather than query parameters and
# errors on them. Hosted databases hand them out in DATABASE_URL anyway.
UNSUPPORTED_QUERY_KEYS = frozenset({"sslmode", "channel_binding", "options"})


def normalise_database_url(url: str) -> str:
    """Rewrite a platform DATABASE_URL into one asyncpg will accept."""
    if not url:
        return url

    parts = urlsplit(url)
    scheme = ASYNC_SCHEME if parts.scheme in ("postgres", "postgresql") else parts.scheme
    query = "&".join(
        piece
        for piece in parts.query.split("&")
        if piece and piece.split("=", 1)[0] not in UNSUPPORTED_QUERY_KEYS
    )
    return urlunsplit((scheme, parts.netloc, parts.path, query, parts.fragment))


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EVALGATE_",
        extra="ignore",
    )

    database_url: str = LOCAL_URL
    test_database_url: str = LOCAL_TEST_URL
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    # Hosts set this one. Used only when EVALGATE_DATABASE_URL is absent.
    platform_database_url: str = Field(default="", validation_alias="DATABASE_URL")
    # Public deployments serve reads only. Nothing should be able to start a
    # run or write a label through an unauthenticated endpoint.
    read_only: bool = False
    db_nullpool: bool = False

    @model_validator(mode="after")
    def _resolve_database_url(self) -> Settings:
        if "EVALGATE_DATABASE_URL" not in os.environ and self.platform_database_url:
            self.database_url = self.platform_database_url
        self.database_url = normalise_database_url(self.database_url)
        self.test_database_url = normalise_database_url(self.test_database_url)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()