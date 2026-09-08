"""Inbound identities: who may call the harness, and with which models.

A principal is one caller — a team, a service, a person. It holds a hashed
inbound key, an optional model allowlist, and an optional per-turn token
ceiling. Principals are loaded from a JSON file at startup and are immutable
for the process lifetime; there is no self-service enrolment endpoint,
deliberately, because minting a credential is an operator action.

Why SHA-256 and not bcrypt/argon2: inbound keys are 256 bits of machine
generated randomness, so there is no dictionary to attack and no need for a
deliberately slow KDF. What matters instead is that a lookup must not leak the
key, which a dict keyed by digest satisfies — recovering the key from a digest
means finding a preimage. Human-chosen passwords would need argon2; these are
not passwords.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import InvalidCredentialError

KEY_PREFIX = "mh_"
_KEY_BYTES = 32


def mint_key() -> str:
    """Generate a fresh inbound key. Shown once, never recoverable."""
    return f"{KEY_PREFIX}{secrets.token_urlsafe(_KEY_BYTES)}"


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Principal:
    id: str
    key_sha256: str = field(repr=False)

    allowed_models: frozenset[str] | None = None
    """``None`` means every model in the catalog. An explicit set restricts."""

    max_tokens_per_turn: int | None = None
    """Upper bound on this principal's `max_tokens`, applied after the model ceiling."""

    allowed_tools: frozenset[str] | None = None
    """Tools this principal may use.

    ``None`` permits the safe tools only. A dangerous tool — one that can
    reach outside this process — must always be named explicitly, so no
    caller acquires side-effecting capability by default. ``"*"`` grants
    everything, including dangerous tools, and is meant for an operator
    principal rather than a shared key.
    """

    disabled: bool = False

    def may_use(self, model_id: str) -> bool:
        return self.allowed_models is None or model_id in self.allowed_models

    def may_use_tool(self, tool_name: str, *, dangerous: bool) -> bool:
        if self.allowed_tools is None:
            # Default grant: safe tools yes, dangerous tools no.
            return not dangerous
        if "*" in self.allowed_tools:
            return True
        return tool_name in self.allowed_tools


ANONYMOUS = Principal(
    id="anonymous",
    key_sha256="",
    allowed_models=None,
)
"""The principal used when inbound auth is explicitly disabled.

Every anonymous caller shares this identity, and therefore shares session
ownership: any of them can read any anonymous session. That is the cost of
running open, and why it takes an explicit flag.
"""


class PrincipalStore:
    def __init__(self, principals: list[Principal]) -> None:
        duplicates = {p.id for p in principals if sum(q.id == p.id for q in principals) > 1}
        if duplicates:
            raise ValueError(f"Duplicate principal id(s): {', '.join(sorted(duplicates))}")

        by_hash: dict[str, Principal] = {}
        for principal in principals:
            if not principal.key_sha256:
                raise ValueError(f"Principal '{principal.id}' has no key_sha256")
            if principal.key_sha256 in by_hash:
                raise ValueError(
                    f"Principals '{by_hash[principal.key_sha256].id}' and "
                    f"'{principal.id}' share a key hash"
                )
            by_hash[principal.key_sha256] = principal

        self._by_hash = by_hash

    def __len__(self) -> int:
        return len(self._by_hash)

    @property
    def ids(self) -> list[str]:
        return sorted(p.id for p in self._by_hash.values())

    def authenticate(self, raw_key: str) -> Principal:
        """Resolve a presented key, or raise.

        A disabled principal and an unknown key raise the same error with the
        same message, so the response cannot be used to enumerate valid keys.
        """
        principal = self._by_hash.get(hash_key(raw_key))
        if principal is None:
            raise InvalidCredentialError("No principal matches the presented key")
        if principal.disabled:
            raise InvalidCredentialError(f"Principal '{principal.id}' is disabled")
        return principal

    # --- loading ----------------------------------------------------------

    @classmethod
    def from_file(cls, path: Path) -> PrincipalStore:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Principals file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Principals file {path} is not valid JSON: {exc}") from exc

        entries = raw.get("principals") if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            # ValueError throughout, not TypeError: every failure in here is a
            # malformed-config error that startup reports as one class.
            raise ValueError(  # noqa: TRY004
                f"Principals file {path} must hold a list, or an object with a 'principals' list"
            )
        return cls([_principal_from_dict(entry, path) for entry in entries])


def _principal_from_dict(entry: object, path: Path) -> Principal:
    if not isinstance(entry, dict):
        raise ValueError(  # noqa: TRY004
            f"Principals file {path}: each entry must be an object"
        )

    missing = {"id", "key_sha256"} - entry.keys()
    if missing:
        raise ValueError(f"Principals file {path}: entry missing {', '.join(sorted(missing))}")

    if "key" in entry:
        # Guard against the obvious mistake of pasting the raw key into the
        # file the server reads. Fail loudly rather than storing a secret.
        raise ValueError(
            f"Principals file {path}: entry '{entry['id']}' contains a raw 'key' "
            "field. Store only 'key_sha256'; mint keys with "
            "`model-harness mint-key`."
        )

    allowed = entry.get("allowed_models")
    if allowed is not None and not isinstance(allowed, list):
        raise ValueError(
            f"Principals file {path}: 'allowed_models' for '{entry['id']}' must be a list"
        )

    allowed_tools = entry.get("allowed_tools")
    if allowed_tools is not None and not isinstance(allowed_tools, list):
        raise ValueError(
            f"Principals file {path}: 'allowed_tools' for '{entry['id']}' must be a list"
        )

    return Principal(
        id=str(entry["id"]),
        key_sha256=str(entry["key_sha256"]),
        allowed_models=frozenset(str(m) for m in allowed) if allowed is not None else None,
        max_tokens_per_turn=entry.get("max_tokens_per_turn"),
        allowed_tools=(
            frozenset(str(t) for t in allowed_tools) if allowed_tools is not None else None
        ),
        disabled=bool(entry.get("disabled", False)),
    )
