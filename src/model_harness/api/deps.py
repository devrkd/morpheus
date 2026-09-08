"""Dependency wiring.

The runner is built once at startup and stashed on ``app.state``, so a request
never pays for client construction. Swapping the session store or the model
factory is a change to :func:`build_runner` alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, Request

from ..config import Settings, get_settings
from ..harness.models import ModelFactory
from ..harness.ownership import OwnershipIndex
from ..harness.runner import AgentRunner


def build_runner(settings: Settings, model_override: Any | None = None) -> AgentRunner:
    """Assemble the harness.

    ``model_override`` exists for tests: a scripted Strands ``Model`` lets the
    whole HTTP surface be exercised with no credential and no spend.
    """
    session_dir = Path(settings.session_dir)
    return AgentRunner(
        settings=settings,
        factory=ModelFactory(settings),
        ownership=OwnershipIndex(session_dir / "ownership.json"),
        session_dir=session_dir / "transcripts",
        model_override=model_override,
    )


def get_runner(request: Request) -> AgentRunner:
    return request.app.state.runner  # type: ignore[no-any-return]


RunnerDep = Annotated[AgentRunner, Depends(get_runner)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
