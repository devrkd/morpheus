"""Harness error hierarchy.

Provider adapters translate their SDK's typed exceptions into these, so a
client sees one error vocabulary regardless of which vendor served (or failed
to serve) the request.

Two audiences, deliberately separated:

* ``message`` is what the caller sees. Errors the harness raises about the
  caller's own request explain themselves fully — that is how a client fixes
  its call. Errors originating inside a provider do **not**: they carry a
  generic message, because the SDK exception behind them embeds provider
  response bodies and internal detail the caller has no business reading.
* ``detail`` is server-only. It is logged against ``error_id``, which the
  caller does see, so a support request quoting that id leads straight to the
  full context in the log.
"""

from __future__ import annotations

from uuid import uuid4


class HarnessError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    retryable: bool = False

    expose_message: bool = True
    """Whether ``message`` is safe to return to the caller."""

    generic_message: str = "The service could not handle the request."
    """Returned in place of ``message`` when :attr:`expose_message` is false."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.detail = detail
        self.error_id = f"err_{uuid4().hex[:12]}"

    def to_payload(self) -> dict[str, object]:
        return {
            "error": {
                "code": self.code,
                "message": self.message if self.expose_message else self.generic_message,
                "provider": self.provider,
                "retryable": self.retryable,
                "error_id": self.error_id,
            }
        }

    def log_line(self) -> str:
        """The full internal record, for the server log only."""
        parts = [f"{self.error_id} {self.code}: {self.message}"]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.detail:
            parts.append(f"detail={self.detail}")
        return " | ".join(parts)


# --- caller-facing: these explain themselves ------------------------------


class UnknownModelError(HarnessError):
    status_code = 404
    code = "unknown_model"


class InvalidRequestError(HarnessError):
    status_code = 400
    code = "invalid_request"


class SessionNotFoundError(HarnessError):
    status_code = 404
    code = "session_not_found"


class ProviderUnavailableError(HarnessError):
    """The provider is known but has no resolvable credential."""

    status_code = 503
    code = "provider_unavailable"


# --- inbound authentication and authorization -----------------------------


class AuthenticationRequiredError(HarnessError):
    status_code = 401
    code = "authentication_required"


class InvalidCredentialError(HarnessError):
    """A presented inbound key did not match any principal, or is disabled.

    Never says which: distinguishing "no such key" from "key disabled" hands
    an attacker a key-enumeration oracle.
    """

    status_code = 401
    code = "invalid_credential"
    expose_message = False
    generic_message = "The supplied API key is not valid."


class ModelNotPermittedError(HarnessError):
    """Authenticated, but this principal may not use this model."""

    status_code = 403
    code = "model_not_permitted"


class ToolNotPermittedError(HarnessError):
    """Authenticated, but this principal may not use this tool."""

    status_code = 403
    code = "tool_not_permitted"


class SpendLimitExceededError(HarnessError):
    status_code = 429
    code = "spend_limit_exceeded"


# --- provider-originated: these stay generic ------------------------------


class ProviderAuthError(HarnessError):
    """The harness's own credential was rejected by the provider.

    Deliberately opaque to the caller: this is an operator problem, and the
    underlying exception can carry provider account detail.
    """

    status_code = 502
    code = "provider_auth_failed"
    expose_message = False
    generic_message = (
        "The service could not authenticate with the upstream model provider. "
        "This is a server-side configuration problem, not a problem with your request."
    )


class ProviderRateLimitError(HarnessError):
    status_code = 429
    code = "provider_rate_limited"
    retryable = True
    expose_message = False
    generic_message = "The upstream model provider is rate limiting this service. Retry shortly."

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        detail: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message, provider=provider, detail=detail)
        self.retry_after = retry_after


class ProviderBadRequestError(HarnessError):
    """The provider rejected the translated request.

    Surfaced as 400 rather than 502 — a rejected translation almost always
    traces back to something in the caller's request the harness could not
    neutralise (an oversized prompt, an unsupported content block). The
    provider's own wording is withheld; ``error_id`` is the way to the detail.
    """

    status_code = 400
    code = "provider_rejected_request"
    expose_message = False
    generic_message = (
        "The upstream model provider rejected this request. Quote the error_id when reporting it."
    )


class ProviderTimeoutError(HarnessError):
    status_code = 504
    code = "provider_timeout"
    retryable = True
    expose_message = False
    generic_message = "The upstream model provider did not respond in time."


class ProviderConnectionError(HarnessError):
    status_code = 502
    code = "provider_unreachable"
    retryable = True
    expose_message = False
    generic_message = "The service could not reach the upstream model provider."


class ProviderServerError(HarnessError):
    status_code = 502
    code = "provider_server_error"
    retryable = True
    expose_message = False
    generic_message = "The upstream model provider returned an error."
