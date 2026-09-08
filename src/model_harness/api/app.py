"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..auth.principals import PrincipalStore
from ..config import Settings, get_settings
from ..errors import HarnessError, ProviderRateLimitError
from .deps import build_runner
from .routes import router

logger = logging.getLogger("model_harness")


class ConfigurationError(RuntimeError):
    """Startup refused because the configuration is unsafe or incoherent."""


def _load_principals(settings: Settings) -> PrincipalStore | None:
    """Resolve inbound auth configuration, failing closed.

    Refusing to start is the right behaviour for a missing principals file.
    The alternative — booting with authentication silently off — turns one
    forgotten environment variable into an open proxy for the provider
    credentials this service holds. Running open stays possible, but only when
    it was asked for.
    """
    if settings.principals_file is not None:
        if settings.allow_anonymous:
            raise ConfigurationError(
                "HARNESS_PRINCIPALS_FILE and HARNESS_ALLOW_ANONYMOUS are both set. "
                "Pick one: authenticated access, or open access."
            )
        try:
            store = PrincipalStore.from_file(settings.principals_file)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        if len(store) == 0:
            raise ConfigurationError(
                f"Principals file {settings.principals_file} defines no principals, "
                "so no caller could authenticate."
            )
        return store

    if settings.allow_anonymous:
        return None

    raise ConfigurationError(
        "Inbound authentication is not configured. Either set "
        "HARNESS_PRINCIPALS_FILE to a principals file (mint a key with "
        "`model-harness mint-key --id <name>`), or set "
        "HARNESS_ALLOW_ANONYMOUS=true to run with no authentication — which "
        "lets anyone who can reach this port spend the provider credentials."
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    app.state.runner = build_runner(settings)

    store: PrincipalStore | None = app.state.principals
    if store is None:
        logger.warning(
            "INBOUND AUTHENTICATION IS DISABLED (HARNESS_ALLOW_ANONYMOUS). Every "
            "caller shares the 'anonymous' principal and can therefore read every "
            "anonymous session. Do not expose this beyond localhost."
        )
    else:
        logger.info(
            "inbound auth enabled; %d principal(s): %s",
            len(store),
            ", ".join(store.ids),
        )

    for name, status in app.state.runner.provider_status().items():
        if status["credential_detected"]:
            logger.info(
                "provider %s: credential via %s (%s)",
                name,
                status["credential_source"],
                status["note"],
            )
        else:
            logger.warning("provider %s: no credential detected — %s", name, status["note"])
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    app = FastAPI(
        title="model-harness",
        version="0.3.0",
        description=(
            "A Bedrock-style unified inference service. One Converse API, one model "
            "id, routed to Anthropic or OpenAI, with server-side conversation "
            "sessions so context survives across turns and across providers."
        ),
        lifespan=_lifespan,
    )
    app.state.settings = resolved
    # Resolved before the app is returned, so a misconfiguration fails at
    # construction rather than on the first request.
    app.state.principals = _load_principals(resolved)

    @app.exception_handler(HarnessError)
    async def _harness_error(request: Request, exc: HarnessError) -> JSONResponse:
        headers = {}
        if isinstance(exc, ProviderRateLimitError) and exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)
        if exc.status_code == 401:
            # RFC 9110: a 401 must say how to authenticate.
            headers["WWW-Authenticate"] = 'Bearer realm="model-harness"'

        # The full detail goes to the log against error_id; the response gets
        # only what the error class considers safe to expose.
        log = logger.error if exc.status_code >= 500 else logger.warning
        log("%s", exc.log_line())

        return JSONResponse(status_code=exc.status_code, content=exc.to_payload(), headers=headers)

    app.include_router(router)

    # The web client is served by this same app, on purpose. A browser page on
    # another origin could not call the API without CORS, and opening CORS on a
    # service that holds provider credentials is a bigger decision than a demo
    # UI warrants. Same origin means no CORS headers, no preflight, and no
    # second process to run.
    #
    # Bearer tokens are also the right fit here: unlike cookies, a browser
    # never attaches an Authorization header on its own, so there is no CSRF
    # surface to defend.
    web_dir = Path(__file__).parent / "web"
    if web_dir.is_dir():
        app.mount("/app", StaticFiles(directory=web_dir, html=True), name="app")

        @app.get("/", include_in_schema=False)
        async def _root() -> RedirectResponse:
            return RedirectResponse(url="/app/")

    return app


# Deliberately no module-level `app`. Building one at import time would run
# the fail-closed configuration check as an import side effect, so a
# misconfiguration would surface as an ImportError from whatever happened to
# import this module first. Serve it as a factory instead:
#
#     uvicorn --factory model_harness.api.app:create_app
