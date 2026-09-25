"""Translating provider and framework failures into our error vocabulary.

Before the migration each adapter mapped its SDK's typed exceptions onto
:mod:`morpheus.errors`. Strands calls the provider SDKs directly, so
without this every provider failure — a rejected key, a rate limit, an
oversized prompt — reached the client as an unexplained 500. That regression
is what turned a missing credential into `Unexpected token 'I'` in the web
client instead of a readable 503.

The two SDKs raise parallel hierarchies, so one table covers both, plus the
Strands-level exceptions and one raw ``TypeError``: the Anthropic SDK signals
"no credential resolved at all" that way, at request time rather than at
construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import anthropic
import openai
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
    ModelThrottledException,
)

from ..errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from .credentials_view import CredentialView

_AUTH = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
)
_RATE = (anthropic.RateLimitError, openai.RateLimitError)
_BAD = (
    anthropic.BadRequestError,
    anthropic.NotFoundError,
    openai.BadRequestError,
    openai.NotFoundError,
)
_TIMEOUT = (anthropic.APITimeoutError, openai.APITimeoutError)
_CONNECT = (anthropic.APIConnectionError, openai.APIConnectionError)
_STATUS = (anthropic.APIStatusError, openai.APIStatusError)


def _retry_after(exc: Any) -> int | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    raw = response.headers.get("retry-after")
    return int(raw) if raw and str(raw).isdigit() else None


@contextmanager
def translated(provider: str, credential: CredentialView) -> Iterator[None]:
    """Map whatever the provider or framework raises onto a harness error."""
    try:
        yield
    except TypeError as exc:
        if credential.detected:
            # A genuine TypeError in our own code must stay a bug, not be
            # relabelled as a configuration problem.
            raise
        # The Anthropic SDK defers credential resolution to request time and
        # reports total failure as a bare TypeError.
        raise ProviderUnavailableError(
            "No credential could be resolved for this provider. Set the API "
            "key, run `ant auth login`, or configure workload identity.",
            provider=provider,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except _AUTH as exc:
        raise ProviderAuthError(
            "The provider rejected the harness credential",
            provider=provider,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except (*_RATE, ModelThrottledException) as exc:
        raise ProviderRateLimitError(
            "The provider is rate limiting this service",
            provider=provider,
            detail=f"{type(exc).__name__}: {exc}",
            retry_after=_retry_after(exc),
        ) from exc
    except ContextWindowOverflowException as exc:
        # Caller-facing on purpose: this one is actionable, and naming the
        # cause is more useful than an opaque error id.
        raise ProviderBadRequestError(
            "This conversation no longer fits in the model's context window. "
            "Start a new session, or use a model with a larger window.",
            provider=provider,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except MaxTokensReachedException as exc:
        raise ProviderBadRequestError(
            "The model hit its output limit before finishing",
            provider=provider,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except _BAD as exc:
        raise ProviderBadRequestError(
            "The provider rejected the request",
            provider=provider,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except _TIMEOUT as exc:
        raise ProviderTimeoutError(
            "The provider did not respond in time",
            provider=provider,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except _STATUS as exc:
        if exc.status_code >= 500:
            raise ProviderServerError(
                "The provider returned a server error",
                provider=provider,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        raise ProviderBadRequestError(
            "The provider rejected the request",
            provider=provider,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except _CONNECT as exc:
        raise ProviderConnectionError(
            "Could not reach the provider",
            provider=provider,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
