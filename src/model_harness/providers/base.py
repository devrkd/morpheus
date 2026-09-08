"""The provider contract every backend adapter implements.

Adding a vendor means writing one class with these four members and adding its
models to the registry. Nothing else in the service changes — the API layer,
the session store and the canonical types are all provider-agnostic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.registry import ModelSpec, effort_rank
from ..core.types import (
    Effort,
    InferenceConfig,
    Message,
    StopReason,
    StreamEvent,
    SystemBlock,
    Usage,
)
from ..tools.base import Tool


@dataclass
class ProviderCall:
    """Everything an adapter needs for one turn, already resolved.

    ``messages`` is the *full* transcript to send — the service has already
    prepended the session history and applied same-provider native replay.
    """

    spec: ModelSpec
    system: list[SystemBlock] | None
    messages: list[Message]
    inference_config: InferenceConfig
    effort: Effort | None
    want_reasoning: bool
    max_tokens: int
    tools: list[Tool] = field(default_factory=list)
    """Tools to expose this turn. Empty means a plain completion."""


@dataclass
class ProviderResult:
    message: Message
    stop_reason: StopReason
    usage: Usage
    native_model: str
    stop_details: dict[str, Any] | None = None
    adjustments: list[str] = field(default_factory=list)


@dataclass
class StreamSink:
    """Out-of-band channel for a streamed turn's final result.

    An adapter's :meth:`LLMProvider.stream` yields display events and, once the
    stream completes, writes the accumulated :class:`ProviderResult` here so
    the service can persist the turn with its native payload intact. Passing a
    per-request sink keeps the adapter itself stateless, which matters because
    one provider instance serves every concurrent request.
    """

    result: ProviderResult | None = None


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def available(self) -> bool:
        """Whether a credential was configured for this provider at startup."""
        ...

    async def converse(self, call: ProviderCall) -> ProviderResult: ...

    def stream(self, call: ProviderCall, sink: StreamSink) -> AsyncIterator[StreamEvent]: ...


# --- shared translation helpers ------------------------------------------


def resolve_effort(effort: Effort | None, spec: ModelSpec) -> tuple[Effort | None, list[str]]:
    """Drop or clamp an effort request to what the target model accepts."""
    if effort is None:
        return None, []
    if not spec.supports_effort:
        return None, [f"dropped effort: not accepted by {spec.id}"]
    if effort_rank(effort) <= effort_rank(spec.max_effort):
        return effort, []
    return spec.max_effort, [
        f"effort '{effort}' clamped to '{spec.max_effort}' (highest supported by {spec.id})"
    ]


def sampling_params(cfg: InferenceConfig, spec: ModelSpec) -> tuple[dict[str, Any], list[str]]:
    """Return the sampling parameters to forward, plus notes for those dropped.

    Dropping rather than forwarding is the whole point: the current Anthropic
    frontier models and OpenAI's reasoning models answer a stray
    ``temperature`` with a 400, so passing a client's harmless default through
    would break requests that ought to succeed.
    """
    requested = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
    }
    present = {k: v for k, v in requested.items() if v is not None}
    if not present:
        return {}, []

    if not spec.supports_sampling:
        names = ", ".join(sorted(present))
        return {}, [f"dropped {names}: not accepted by {spec.id}"]

    return present, []
