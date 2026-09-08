"""Service configuration, read from the environment (and a local .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- outbound provider credentials -----------------------------------
    # All optional. Leaving one unset does not disable its provider: the SDK
    # walks its own credential chain (env key, auth token, `ant auth login`
    # profile, workload identity federation). These settings only override it.
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_profile: str | None = Field(default=None, alias="ANTHROPIC_PROFILE")
    anthropic_base_url: str | None = Field(default=None, alias="ANTHROPIC_BASE_URL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_organization: str | None = Field(default=None, alias="OPENAI_ORG_ID")
    openai_project: str | None = Field(default=None, alias="OPENAI_PROJECT_ID")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    # --- inbound authentication ------------------------------------------
    principals_file: Path | None = Field(default=None, alias="HARNESS_PRINCIPALS_FILE")
    """JSON file of inbound principals. Required unless anonymous access is allowed."""

    allow_anonymous: bool = Field(default=False, alias="HARNESS_ALLOW_ANONYMOUS")
    """Run with no inbound authentication.

    Off by default, and startup fails when neither this nor a principals file
    is set. An unauthenticated harness lets anyone who can reach the port spend
    the provider credentials, so it has to be asked for explicitly rather than
    being what you get by forgetting to configure something.
    """

    # --- service ----------------------------------------------------------
    host: str = Field(default="127.0.0.1", alias="HARNESS_HOST")
    port: int = Field(default=8080, alias="HARNESS_PORT")

    session_dir: str = Field(default="./.sessions", alias="HARNESS_SESSION_DIR")
    """Where transcripts and the ownership index live.

    Strands persists conversations here. A multi-instance deployment wants a
    shared backend instead — `S3SessionManager` for transcripts, and the
    ownership index moved to Redis or a database.
    """

    mcp_servers: str | None = Field(default=None, alias="HARNESS_MCP_SERVERS")
    """JSON list of MCP servers to expose as tools."""

    session_ttl_seconds: int = Field(default=86_400, alias="HARNESS_SESSION_TTL_SECONDS")
    session_max_turns: int = Field(default=200, alias="HARNESS_SESSION_MAX_TURNS")
    request_timeout_seconds: float = Field(default=600.0, alias="HARNESS_REQUEST_TIMEOUT_SECONDS")

    # Default output ceilings. Streaming gets the larger one because HTTP
    # timeouts are not a concern there; a non-streaming request with a huge
    # max_tokens risks the connection dropping before the response lands.
    default_max_tokens: int = 16_000
    default_max_tokens_streaming: int = 64_000


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
