"""Session ownership — the part of session handling that stays ours.

Strands owns the transcript: ``FileSessionManager`` and ``S3SessionManager``
persist messages, restore them into a fresh ``Agent``, and manage the
conversation window. That is a better implementation than ours, and it goes.

What Strands has no concept of is *who a session belongs to*. It takes a
session id and loads it. Left at that, any caller who learned or guessed an id
could read another principal's transcript — the sharpest hole we closed
earlier, and it would reopen on migration.

So authorization stays on our side of the boundary: a small index from session
id to owning principal, consulted before Strands is ever handed the id. The
division is deliberate — Strands stores conversations, we decide who may open
one.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class SessionRecord:
    session_id: str
    owner: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    turn_count: int = 0
    last_provider: str | None = None
    last_model: str | None = None
    preview: str = ""

    def summary(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "owner": self.owner,
            "preview": self.preview,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": self.turn_count,
            "last_provider": self.last_provider,
            "last_model": self.last_model,
        }


class OwnershipIndex:
    """Session id -> owner, plus the metadata a session list needs.

    Persisted as one JSON file alongside the transcripts so it survives a
    restart, which the in-memory store did not. Reads and writes are guarded
    by an asyncio lock; a multi-process deployment wants this in Redis or a
    database, and the interface is narrow enough to make that a drop-in.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._records: dict[str, SessionRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt index must not stop the service booting; sessions
            # become unreachable rather than the process failing to start.
            return
        for entry in raw.get("sessions", []):
            record = SessionRecord(**entry)
            self._records[record.session_id] = record

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sessions": [r.__dict__ for r in self._records.values()]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)  # atomic, so a crash mid-write cannot corrupt it

    # --- API --------------------------------------------------------------

    async def claim(self, session_id: str | None, owner: str) -> SessionRecord:
        """Get the caller's session, or create one they own.

        An id belonging to someone else raises, rather than being adopted:
        writing into a live session owned by another principal would leak this
        turn to them and their history to this caller.
        """
        async with self._lock:
            sid = session_id or f"sess_{uuid.uuid4().hex[:24]}"
            existing = self._records.get(sid)
            if existing is not None:
                if existing.owner != owner:
                    raise PermissionError(sid)
                return existing

            record = SessionRecord(session_id=sid, owner=owner)
            self._records[sid] = record
            self._flush()
            return record

    async def get(self, session_id: str, owner: str) -> SessionRecord | None:
        """Read a session, or None if it is absent *or* someone else's.

        Absent and forbidden are deliberately indistinguishable: a 403 would
        confirm the id exists, which is what someone probing for other
        callers' sessions wants to learn.
        """
        async with self._lock:
            record = self._records.get(session_id)
            if record is None or record.owner != owner:
                return None
            return record

    async def record_turn(
        self,
        session_id: str,
        *,
        provider: str,
        model: str,
        preview: str | None = None,
    ) -> None:
        async with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return
            record.turn_count += 1
            record.last_provider = provider
            record.last_model = model
            record.updated_at = datetime.now(UTC).isoformat()
            if preview and not record.preview:
                record.preview = preview[:80]
            self._flush()

    async def discard_if_unused(self, session_id: str) -> None:
        """Drop a session that never recorded a turn.

        A session is claimed before the model is called, so a failed first
        request would otherwise leave an empty one in the caller's list.
        """
        async with self._lock:
            record = self._records.get(session_id)
            if record is not None and record.turn_count == 0:
                del self._records[session_id]
                self._flush()

    async def delete(self, session_id: str, owner: str) -> bool:
        async with self._lock:
            record = self._records.get(session_id)
            if record is None or record.owner != owner:
                return False
            del self._records[session_id]
            self._flush()
            return True

    async def list_for(self, owner: str, limit: int = 100) -> list[SessionRecord]:
        async with self._lock:
            mine = [r for r in self._records.values() if r.owner == owner]
            mine.sort(key=lambda r: r.updated_at, reverse=True)
            return mine[:limit]
