"""Outbound credential resolution.

The first version of this service read one environment variable per provider
and refused to build a client without it. That was wrong in both directions:
it could not use the credential mechanisms a server actually wants (OAuth
profiles, workload identity federation), and it turned "I could not detect a
key" into "this provider cannot be used" — a 503 for a request that would in
fact have authenticated.

The rule now is: let the SDK resolve the credential, because each SDK
implements a documented chain that is richer than one env var. Detection here
is only for reporting — ``GET /v1/health`` and ``GET /v1/models`` say which
source was found — and an undetected credential does not block the request. If
nothing resolves, the provider answers with an auth error, which surfaces as a
502 naming a configuration problem rather than a silent 503.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class CredentialSource(StrEnum):
    API_KEY = "api_key"
    AUTH_TOKEN = "auth_token"
    OAUTH_PROFILE = "oauth_profile"
    WORKLOAD_IDENTITY = "workload_identity"
    UNDETECTED = "undetected"

    @property
    def detected(self) -> bool:
        return self is not CredentialSource.UNDETECTED


@dataclass(frozen=True)
class CredentialInfo:
    source: CredentialSource
    note: str = ""

    @property
    def detected(self) -> bool:
        return self.source.detected


_ANTHROPIC_WIF_VARS = (
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
)


def detect_anthropic(
    *,
    configured_api_key: str | None,
    configured_profile: str | None,
    env: dict[str, str] | None = None,
    config_dir: Path | None = None,
) -> CredentialInfo:
    """Report which link of the Anthropic SDK's chain will be used.

    Mirrors the SDK's documented precedence: an explicit API key, then
    ``ANTHROPIC_API_KEY``, then ``ANTHROPIC_AUTH_TOKEN``, then a named or
    active OAuth profile from ``ant auth login``, then workload identity
    federation, then a default profile on disk.

    Detection is best-effort by design. ``UNDETECTED`` means "nothing found
    here", not "authentication will fail" — the SDK may still resolve
    something this function does not know how to look for.
    """
    env = os.environ if env is None else env  # type: ignore[assignment]

    if configured_api_key:
        return CredentialInfo(CredentialSource.API_KEY, "explicitly configured")
    if env.get("ANTHROPIC_API_KEY"):
        return CredentialInfo(CredentialSource.API_KEY, "ANTHROPIC_API_KEY")
    if env.get("ANTHROPIC_AUTH_TOKEN"):
        return CredentialInfo(CredentialSource.AUTH_TOKEN, "ANTHROPIC_AUTH_TOKEN")

    profile = configured_profile or env.get("ANTHROPIC_PROFILE")
    if profile:
        return CredentialInfo(CredentialSource.OAUTH_PROFILE, f"profile '{profile}'")

    if all(env.get(var) for var in _ANTHROPIC_WIF_VARS) and (
        env.get("ANTHROPIC_IDENTITY_TOKEN_FILE") or env.get("ANTHROPIC_IDENTITY_TOKEN")
    ):
        return CredentialInfo(CredentialSource.WORKLOAD_IDENTITY, "federation env vars present")

    directory = config_dir or Path.home() / ".config" / "anthropic"
    if directory.is_dir() and any(directory.iterdir()):
        return CredentialInfo(CredentialSource.OAUTH_PROFILE, f"default profile in {directory}")

    return CredentialInfo(
        CredentialSource.UNDETECTED,
        "set ANTHROPIC_API_KEY, or run `ant auth login`, or configure workload identity federation",
    )


def detect_openai(
    *,
    configured_api_key: str | None,
    env: dict[str, str] | None = None,
) -> CredentialInfo:
    """Report which credential the OpenAI SDK will use.

    Narrower than the Anthropic chain: an API key, or workload identity via
    the SDK's own env vars. The SDK raises at construction time when it finds
    neither, so the adapter treats a construction failure as authoritative.
    """
    env = os.environ if env is None else env  # type: ignore[assignment]

    if configured_api_key:
        return CredentialInfo(CredentialSource.API_KEY, "explicitly configured")
    if env.get("OPENAI_API_KEY"):
        return CredentialInfo(CredentialSource.API_KEY, "OPENAI_API_KEY")
    if env.get("OPENAI_WORKLOAD_IDENTITY_TOKEN") or env.get("OPENAI_WORKLOAD_IDENTITY_TOKEN_FILE"):
        return CredentialInfo(CredentialSource.WORKLOAD_IDENTITY, "workload identity env")

    return CredentialInfo(
        CredentialSource.UNDETECTED, "set OPENAI_API_KEY or configure workload identity"
    )
