"""Layer 2: one turn, run through a Strands agent.

This replaces ``core/service.py``'s hand-written tool loop. What it still does
is everything Strands has no opinion about, because it is our policy rather
than agent mechanics:

* resolve a model id through our catalog and check the principal's allowlist;
* resolve tool names and check the principal's tool grants;
* claim the session under its owner before Strands ever sees the id;
* translate a Strands ``AgentResult`` back into our canonical response.

The loop, the tool execution, the schema generation, the session file format
and the provider wire formats are all Strands'. That is the point of the
migration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from strands import Agent
from strands.session import FileSessionManager

from ..auth.principals import Principal
from ..config import Settings
from ..core.registry import ModelSpec
from ..core.registry import resolve as resolve_model
from ..core.types import (
    ConverseOutput,
    ConverseRequest,
    ConverseResponse,
    Message,
    ReasoningBlock,
    Role,
    StopReason,
    TextBlock,
    ToolInvocation,
    Usage,
)
from ..errors import (
    InvalidRequestError,
    ModelNotPermittedError,
    SessionNotFoundError,
    UnknownModelError,
)
from . import tools as tool_layer
from .models import ModelFactory
from .ownership import OwnershipIndex

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "tool_use": StopReason.TOOL_USE,
    "refusal": StopReason.REFUSAL,
    "guardrail_intervened": StopReason.REFUSAL,
}


class AgentRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        factory: ModelFactory,
        ownership: OwnershipIndex,
        session_dir: Path,
        model_override: Any | None = None,
    ) -> None:
        self._settings = settings
        self._factory = factory
        self._ownership = ownership
        self._session_dir = session_dir
        # Tests inject a scripted Model here so the loop can be driven with no
        # credential and no spend.
        self._model_override = model_override

    # --- policy -----------------------------------------------------------

    def _route(self, model_id: str, principal: Principal) -> ModelSpec:
        spec = resolve_model(model_id)
        if spec is None:
            raise UnknownModelError(
                f"Unknown model '{model_id}'. See GET /v1/models for the catalog."
            )
        # Checked against the canonical id, so an alias cannot slip past an
        # allowlist that omits the model it resolves to.
        if not principal.may_use(spec.id):
            raise ModelNotPermittedError(
                f"Principal '{principal.id}' is not permitted to use model '{spec.id}'"
            )
        return spec

    @staticmethod
    def _prompt_from(request: ConverseRequest) -> str:
        last = request.messages[-1]
        if last.role is not Role.USER:
            raise InvalidRequestError(
                "The last message of a request must have role 'user' — send only "
                "the new turn; the stored transcript is prepended for you"
            )
        text = "\n\n".join(m.text() for m in request.messages if m.text())
        if not text:
            raise InvalidRequestError("The request contains no text to send")
        return text

    # --- execution --------------------------------------------------------

    async def converse(self, request: ConverseRequest, principal: Principal) -> ConverseResponse:
        spec = self._route(request.model, principal)
        prompt = self._prompt_from(request)
        granted = tool_layer.resolve(request.tools, principal)

        try:
            record = await self._ownership.claim(request.session_id, principal.id)
        except PermissionError as exc:
            raise SessionNotFoundError(f"No session '{exc.args[0]}'") from exc

        adjustments: list[str] = []
        model = self._model_override
        if model is None:
            model, notes = self._factory.build_with_fallbacks(spec, [])
            adjustments.extend(notes)

        system = "\n\n".join(b.text for b in request.system) if request.system else None

        agent = Agent(
            model=model,
            tools=granted,
            system_prompt=system,
            # Strands persists and restores the transcript; we only ever hand
            # it an id we have already authorized.
            session_manager=FileSessionManager(
                session_id=record.session_id,
                storage_dir=str(self._session_dir),
            ),
        )

        started = time.perf_counter()
        try:
            result = await agent.invoke_async(prompt)
        except Exception:
            await self._ownership.discard_if_unused(record.session_id)
            raise
        latency_ms = int((time.perf_counter() - started) * 1000)

        await self._ownership.record_turn(
            record.session_id,
            provider=spec.provider.value,
            model=spec.id,
            preview=" ".join(prompt.split()),
        )

        return ConverseResponse(
            session_id=record.session_id,
            provider=spec.provider.value,
            model=spec.id,
            native_model=spec.native_id,
            output=ConverseOutput(message=_to_canonical(result.message)),
            stop_reason=_STOP_REASONS.get(result.stop_reason or "", StopReason.END_TURN),
            usage=_to_usage(result.metrics),
            latency_ms=latency_ms,
            tool_calls=_to_invocations(result.metrics),
            iterations=getattr(result.metrics, "cycle_count", 1) or 1,
            adjustments=adjustments,
        )


# --- translation back to our canonical shape -----------------------------


def _to_canonical(message: dict[str, Any] | None) -> Message:
    """Strands' message dict -> our canonical Message.

    Strands already normalises both providers into one content-block shape, so
    this is a rename rather than the per-provider translation we used to own.
    """
    blocks: list[TextBlock | ReasoningBlock] = []
    for block in (message or {}).get("content", []):
        if "text" in block:
            blocks.append(TextBlock(text=block["text"]))
        elif "reasoningContent" in block:
            reasoning = block["reasoningContent"].get("reasoningText", {})
            blocks.append(ReasoningBlock(text=reasoning.get("text", "")))
    if not blocks:
        blocks.append(TextBlock(text=""))
    return Message(role=Role.ASSISTANT, content=blocks)


def _to_usage(metrics: Any) -> Usage:
    """Read accumulated usage across every loop iteration.

    Strands sums usage over the whole loop, which is what we had to do by hand
    — and it normalises the field names, which is where our own
    cross-provider accounting bug lived. Whether its normalisation gets the
    Anthropic/OpenAI cache-token semantics right still needs checking against
    a live call; the fields are read defensively until then.
    """
    raw = getattr(metrics, "accumulated_usage", None) or {}
    return Usage(
        input_tokens=raw.get("inputTokens", 0) or 0,
        output_tokens=raw.get("outputTokens", 0) or 0,
        cache_read_tokens=raw.get("cacheReadInputTokens", 0) or 0,
        cache_write_tokens=raw.get("cacheWriteInputTokens", 0) or 0,
    )


def _to_invocations(metrics: Any) -> list[ToolInvocation]:
    """Report which tools ran, from Strands' per-tool metrics."""
    invocations: list[ToolInvocation] = []
    for name, tm in (getattr(metrics, "tool_metrics", None) or {}).items():
        calls = getattr(tm, "call_count", 0) or 0
        if not calls:
            continue
        errors = getattr(tm, "error_count", 0) or 0
        seconds = getattr(tm, "total_time", 0.0) or 0.0
        invocations.append(
            ToolInvocation(
                name=name,
                tool_use_id=f"{name}:{calls}",
                input={"calls": calls},
                output=f"{calls} call(s), {errors} error(s)",
                is_error=errors > 0,
                duration_ms=int(seconds * 1000),
            )
        )
    return invocations
