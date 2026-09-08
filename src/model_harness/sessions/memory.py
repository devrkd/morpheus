"""In-process session store.

Good enough for a single instance. It is not shared across workers, so run
uvicorn with one worker while this is the store — otherwise a client's second
turn can land on a process that has never seen its session. Swapping in a
Redis-backed :class:`~model_harness.sessions.base.SessionStore` is the fix, and
the only thing that has to change is which store the app wires up.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

from ..core.types import Role
from .base import Session


class InMemorySessionStore:
    def __init__(self, *, ttl_seconds: int = 86_400, max_turns: int = 200) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._ttl = timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None
        self._max_turns = max_turns

    # --- SessionStore -----------------------------------------------------

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if self._expired(session):
                del self._sessions[session_id]
                return None
            return session

    async def create(self, session_id: str | None = None, owner: str = "anonymous") -> Session:
        sid = session_id or f"sess_{uuid.uuid4().hex[:24]}"
        async with self._lock:
            self._evict_expired()
            session = Session(id=sid, owner=owner)
            self._sessions[sid] = session
            return session

    async def save(self, session: Session) -> None:
        session.updated_at = datetime.now(UTC)
        self._trim(session)
        async with self._lock:
            self._sessions[session.id] = session

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def list_ids(self, limit: int = 100, owner: str | None = None) -> list[str]:
        async with self._lock:
            self._evict_expired()
            visible = [s for s in self._sessions.values() if owner is None or s.owner == owner]
            ordered = sorted(visible, key=lambda s: s.updated_at, reverse=True)
            return [s.id for s in ordered[:limit]]

    # --- internals --------------------------------------------------------

    def _expired(self, session: Session) -> bool:
        if self._ttl is None:
            return False
        return datetime.now(UTC) - session.updated_at > self._ttl

    def _evict_expired(self) -> None:
        if self._ttl is None:
            return
        for sid in [s.id for s in self._sessions.values() if self._expired(s)]:
            del self._sessions[sid]

    def _trim(self, session: Session) -> None:
        """Drop the oldest turns once the session exceeds its cap.

        Trimming starts from the front and always resumes at a user message, so
        the transcript handed to a provider never begins mid-exchange — both
        providers require the first message to be from the user.
        """
        if self._max_turns <= 0:
            return
        limit = self._max_turns * 2
        if len(session.messages) <= limit:
            return

        cut = len(session.messages) - limit
        while cut < len(session.messages) and session.messages[cut].role is not Role.USER:
            cut += 1
        session.messages = session.messages[cut:]
