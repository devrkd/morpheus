"""Session store contract.

The store owns the canonical transcript. It is intentionally a narrow protocol
so an in-memory implementation can be swapped for Redis or a database without
the service layer changing: three reads, one write, one delete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from ..core.types import Message, Role, SystemBlock


@dataclass
class Session:
    id: str
    owner: str = "anonymous"
    """Principal id that created the session.

    Checked on every access. Session ids travel freely between client and
    server, so without an owner any caller who learned or guessed an id could
    read someone else's transcript.
    """

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    system: list[SystemBlock] | None = None
    messages: list[Message] = field(default_factory=list)

    last_provider: str | None = None
    last_model: str | None = None

    turn_count: int = 0
    """Number of completed request/response cycles, not messages."""

    def preview(self, limit: int = 80) -> str:
        """First user turn, truncated — a label for a session list."""
        for message in self.messages:
            if message.role is Role.USER:
                text = " ".join(message.text().split())
                if text:
                    return text if len(text) <= limit else text[: limit - 1] + "…"
        return ""

    def summary(self) -> dict[str, object]:
        """Metadata view, safe to return without dumping the whole transcript."""
        return {
            "session_id": self.id,
            "owner": self.owner,
            "preview": self.preview(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages),
            "turn_count": self.turn_count,
            "last_provider": self.last_provider,
            "last_model": self.last_model,
            "has_system_prompt": self.system is not None,
        }


@runtime_checkable
class SessionStore(Protocol):
    async def get(self, session_id: str) -> Session | None: ...

    async def create(self, session_id: str | None = None, owner: str = "anonymous") -> Session: ...

    async def save(self, session: Session) -> None: ...

    async def delete(self, session_id: str) -> bool: ...

    async def list_ids(self, limit: int = 100, owner: str | None = None) -> list[str]: ...
