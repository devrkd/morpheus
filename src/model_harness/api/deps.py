"""Dependency wiring.

The service is built once at startup and stashed on ``app.state`` so a request
never pays for client construction. Swapping the session store for a shared
one (Redis, a database) is a change to :func:`build_service` alone.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ..config import Settings, get_settings
from ..core.service import ConverseService, build_providers
from ..sessions.memory import InMemorySessionStore
from ..tools.builtin import default_registry


def build_service(settings: Settings) -> ConverseService:
    store = InMemorySessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_turns=settings.session_max_turns,
    )
    return ConverseService(
        providers=build_providers(settings),
        store=store,
        settings=settings,
        tools=default_registry(),
    )


def get_service(request: Request) -> ConverseService:
    return request.app.state.service  # type: ignore[no-any-return]


ServiceDep = Annotated[ConverseService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
