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

import asyncio
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from strands import Agent
from strands.session import FileSessionManager
from strands.types.exceptions import SessionException

from ..auth.principals import Principal
from ..config import Settings
from ..core.registry import ModelSpec, Provider
from ..core.registry import resolve as resolve_model
from ..core.types import (
    ConverseOutput,
    ConverseRequest,
    ConverseResponse,
    Message,
    ReasoningBlock,
    Role,
    StopReason,
    StreamEvent,
    TextBlock,
    ToolInvocation,
    Usage,
)
from ..errors import (
    InvalidRequestError,
    ModelNotPermittedError,
    ProviderTimeoutError,
    SessionNotFoundError,
    UnknownModelError,
)
from .errors import translated
from .models import ModelFactory
from .ownership import OwnershipIndex
from .tools import ToolCatalog

_DEFAULT_AGENT_ID = "default"
"""Strands' default agent id.

It is the *id*, not the directory name: on disk the path is
``agents/agent_<id>``, so passing "agent_default" here silently looks for
``agent_agent_default`` and reports the transcript as missing.
"""

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "tool_use": StopReason.TOOL_USE,
    "refusal": StopReason.REFUSAL,
    "guardrail_intervened": StopReason.REFUSAL,
}


class AgentRunner:
    DEFAULT_TURNS = 5
    """Loop iterations allowed when a request does not say.

    Strands treats an omitted cap as *no limit*, so leaving `limits` unset
    lets a model that keeps calling tools run until something else gives out —
    which is what an apparently stuck request looks like from the outside.
    Uncapped is also a runaway bill: every iteration resends the transcript
    plus every tool result so far.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        factory: ModelFactory,
        ownership: OwnershipIndex,
        session_dir: Path,
        catalog: ToolCatalog | None = None,
        model_override: Any | None = None,
    ) -> None:
        self._settings = settings
        self._factory = factory
        self._ownership = ownership
        self._session_dir = session_dir
        self._catalog = catalog or ToolCatalog()
        # Tests inject a scripted Model here so the loop can be driven with no
        # credential and no spend.
        self._model_override = model_override

    # --- session access (authorization ours, storage Strands') ------------

    @property
    def catalog(self) -> ToolCatalog:
        return self._catalog

    def provider_status(self) -> dict[str, dict[str, object]]:
        return self._factory.status()

    def mcp_status(self) -> list[dict[str, Any]]:
        """Per-server MCP status, for an authenticated caller."""
        return self._catalog.mcp_status()

    def mcp_counts(self) -> dict[str, Any]:
        """Aggregate MCP counts, safe for the open health route."""
        return self._catalog.mcp_counts()

    async def list_sessions(self, principal: Principal, limit: int = 100):
        return await self._ownership.list_for(principal.id, limit)

    async def get_session(self, session_id: str, principal: Principal):
        return await self._ownership.get(session_id, principal.id)

    async def delete_session(self, session_id: str, principal: Principal) -> bool:
        """Forget a session.

        Ownership is dropped first: if the transcript removal fails, the
        session is already unreachable rather than briefly readable by the
        person who asked for it to be gone.
        """
        if not await self._ownership.delete(session_id, principal.id):
            return False
        target = self._session_dir / f"session_{session_id}"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        return True

    async def read_transcript(self, session_id: str) -> list[dict[str, Any]]:
        """Read the stored messages for a session.

        Goes through Strands' session repository rather than reading its file
        layout directly, so the on-disk format stays theirs to change.
        Callers must have authorized the id first — this does not check.
        """
        manager = FileSessionManager(session_id=session_id, storage_dir=str(self._session_dir))
        try:
            stored = manager.list_messages(session_id, _DEFAULT_AGENT_ID)
        except (OSError, KeyError, SessionException):
            # A session claimed but never written has no message directory
            # yet, which Strands reports as a SessionException rather than as
            # a missing-file error.
            return []
        return [m.message for m in stored]

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

    def _limits(self, request: ConverseRequest) -> dict[str, int]:
        """Per-invocation budget for the agent loop.

        Capped at the top of each iteration, so tools requested by the
        previous turn always finish first and the transcript is left
        reinvokable — the invariant we used to hand-build.
        """
        return {"turns": request.max_tool_iterations or self.DEFAULT_TURNS}

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
        granted = self._catalog.resolve(request.tools, principal)

        try:
            record = await self._ownership.claim(request.session_id, principal.id)
        except PermissionError as exc:
            raise SessionNotFoundError(f"No session '{exc.args[0]}'") from exc

        agent, adjustments = self._build_agent(request, record.session_id, spec, granted, principal)

        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._settings.turn_timeout_seconds):
                with translated(spec.provider.value, self._factory.credential_for(spec)):
                    result = await agent.invoke_async(prompt, limits=self._limits(request))
        except TimeoutError as exc:
            await self._ownership.discard_if_unused(record.session_id)
            raise ProviderTimeoutError(
                f"The turn did not finish within "
                f"{self._settings.turn_timeout_seconds:.0f}s and was cancelled.",
                provider=spec.provider.value,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
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

    def _build_agent(
        self,
        request: ConverseRequest,
        session_id: str,
        spec: ModelSpec,
        granted: list[Any],
        principal: Principal,
    ) -> tuple[Agent, list[str]]:
        """Construct the agent for one turn. Shared by buffered and streaming."""
        adjustments: list[str] = []
        sampling, notes = _sampling_params(request, spec)
        adjustments.extend(notes)
        adjustments.extend(_parameter_notes(request, spec, principal))

        model = self._model_override
        if model is None:
            model, chain_notes = self._factory.build_with_fallbacks(
                spec,
                [],
                max_tokens=_resolved_max_tokens(request, spec, principal),
                sampling=sampling or None,
                stop_sequences=request.inference_config.stop_sequences,
            )
            adjustments.extend(chain_notes)

        system = "\n\n".join(b.text for b in request.system) if request.system else None

        return (
            Agent(
                model=model,
                tools=granted,
                system_prompt=system,
                # Strands persists and restores the transcript; we only ever
                # hand it an id we have already authorized.
                session_manager=FileSessionManager(
                    session_id=session_id, storage_dir=str(self._session_dir)
                ),
            ),
            adjustments,
        )

    async def converse_stream(
        self, request: ConverseRequest, principal: Principal
    ) -> tuple[str, AsyncIterator[StreamEvent]]:
        """Stream a turn, tool calls included.

        Tools used to be refused here with a 400, because the old
        implementation piped a single provider response straight through and a
        tool loop is several provider calls. Strands' ``stream_async`` yields
        one continuous stream spanning the whole loop, so that restriction was
        an artefact of our implementation, not a property of the problem.

        The session id is returned before the generator so the route can set it
        as a header ahead of the first body byte.
        """
        spec = self._route(request.model, principal)
        prompt = self._prompt_from(request)
        granted = self._catalog.resolve(request.tools, principal)

        try:
            record = await self._ownership.claim(request.session_id, principal.id)
        except PermissionError as exc:
            raise SessionNotFoundError(f"No session '{exc.args[0]}'") from exc

        agent, adjustments = self._build_agent(request, record.session_id, spec, granted, principal)

        async def generate() -> AsyncIterator[StreamEvent]:
            text_open = False
            announced: set[str] = set()
            names: dict[str, str] = {}
            final: Any = None

            # Every yield sits inside this try. A client hang-up arrives as
            # GeneratorExit or CancelledError at whichever yield is currently
            # suspended — including the very first one — so a guard wrapped
            # only around the model loop would miss the common case.
            try:
                yield StreamEvent(
                    type="message_start",
                    session_id=record.session_id,
                    provider=spec.provider.value,
                    model=spec.id,
                    native_model=spec.native_id,
                    adjustments=adjustments,
                )

                stream = agent.stream_async(prompt, limits=self._limits(request))

                # Wraps the *iteration*, not just the generator's creation:
                # stream_async is lazy, so every provider failure surfaces
                # while consuming it.
                with translated(spec.provider.value, self._factory.credential_for(spec)):
                    async for event in stream:
                        if "data" in event:
                            if not text_open:
                                text_open = True
                                yield StreamEvent(
                                    type="content_block_start", index=0, block_type="text"
                                )
                            yield StreamEvent(type="content_delta", index=0, text=event["data"])

                        elif "current_tool_use" in event:
                            use = event["current_tool_use"] or {}
                            tuid = use.get("toolUseId")
                            # Repeats while the arguments accumulate; announce once.
                            if tuid and tuid not in announced:
                                announced.add(tuid)
                                names[tuid] = use.get("name", "")
                                yield StreamEvent(
                                    type="tool_start",
                                    tool_use_id=tuid,
                                    tool_name=use.get("name"),
                                )

                        elif "message" in event:
                            for block in (event["message"] or {}).get("content", []):
                                result = block.get("toolResult")
                                if not result:
                                    continue
                                tuid = result.get("toolUseId", "")
                                yield StreamEvent(
                                    type="tool_end",
                                    tool_use_id=tuid,
                                    tool_name=names.get(tuid),
                                    is_error=result.get("status") == "error",
                                    tool_output=(
                                        _truncate_for_stream(_tool_result_text(result))
                                        if request.stream_tool_output
                                        else None
                                    ),
                                )

                        elif "result" in event:
                            final = event["result"]

                if text_open:
                    yield StreamEvent(type="content_block_stop", index=0)

                await self._ownership.record_turn(
                    record.session_id,
                    provider=spec.provider.value,
                    model=spec.id,
                    preview=" ".join(prompt.split()),
                )

                metrics = getattr(final, "metrics", None)
                yield StreamEvent(
                    type="message_stop",
                    session_id=record.session_id,
                    provider=spec.provider.value,
                    model=spec.id,
                    native_model=spec.native_id,
                    stop_reason=_STOP_REASONS.get(
                        getattr(final, "stop_reason", "") or "", StopReason.END_TURN
                    ),
                    usage=_to_usage(metrics),
                    iterations=getattr(metrics, "cycle_count", 1) or 1,
                )
            except BaseException:
                # Strands has already persisted every message produced so far
                # — each tool call with its matching result — so a mid-loop
                # hang-up leaves a replayable transcript and there is nothing
                # to repair. Only a turn that never completed needs its empty
                # session removed, so it does not litter the caller's list.
                await self._ownership.discard_if_unused(record.session_id)
                raise

        return record.session_id, generate()


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


_STREAM_OUTPUT_LIMIT = 4_000


def _truncate_for_stream(text: str) -> str:
    """Cap tool output pushed down the SSE channel.

    A tool result can be hundreds of kilobytes. The full value is already in
    the persisted transcript, and a client rendering progress does not need it
    inline — which is why `stream_tool_output` is off by default.
    """
    if len(text) <= _STREAM_OUTPUT_LIMIT:
        return text
    return text[:_STREAM_OUTPUT_LIMIT] + "… [truncated in stream]"


def _tool_result_text(result: dict[str, Any]) -> str:
    """Flatten a Strands toolResult's content blocks into plain text."""
    parts = []
    for block in result.get("content", []) or []:
        if "text" in block:
            parts.append(block["text"])
        elif "json" in block:
            parts.append(str(block["json"]))
    return "\n".join(parts)


def _sampling_params(request: ConverseRequest, spec: ModelSpec) -> tuple[dict[str, Any], list[str]]:
    """Sampling parameters to forward, and notes for those dropped.

    Dropping rather than forwarding is the point: the current Anthropic
    frontier models and OpenAI's reasoning models answer a stray
    ``temperature`` with a 400, so passing a client's harmless default through
    would break a request that ought to succeed.
    """
    cfg = request.inference_config
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

    # top_k is Anthropic-only; OpenAI has no equivalent.
    notes: list[str] = []
    if spec.provider is not Provider.ANTHROPIC and "top_k" in present:
        del present["top_k"]
        notes.append("dropped top_k: no equivalent in the OpenAI API")
    return present, notes


def _resolved_max_tokens(request: ConverseRequest, spec: ModelSpec, principal: Principal) -> int:
    """The output ceiling for this turn: request, model cap, principal cap."""
    requested = request.inference_config.max_tokens or 0
    ceilings = [c for c in (requested, spec.max_output_tokens, principal.max_tokens_per_turn) if c]
    return min(ceilings) if ceilings else spec.max_output_tokens


def _parameter_notes(request: ConverseRequest, spec: ModelSpec, principal: Principal) -> list[str]:
    """Report every neutral parameter that will not reach the model verbatim.

    This is the one contract the service has kept from the start: nothing is
    altered without saying so. A caller who set `effort` and got a different
    answer than expected deserves to know it never left the building.
    """
    notes: list[str] = []
    cfg = request.inference_config

    requested = cfg.max_tokens or 0
    if requested and requested > spec.max_output_tokens:
        notes.append(
            f"max_tokens {requested} lowered to {spec.max_output_tokens}, "
            f"the output ceiling for {spec.id}"
        )
    if principal.max_tokens_per_turn is not None:
        effective = _resolved_max_tokens(request, spec, principal)
        if effective == principal.max_tokens_per_turn and (
            not requested or requested > principal.max_tokens_per_turn
        ):
            notes.append(
                f"max_tokens capped at {principal.max_tokens_per_turn}, the "
                f"per-turn ceiling for principal '{principal.id}'"
            )

    if request.effort is not None:
        # Anthropic exposes this as output_config.effort and OpenAI as
        # reasoning_effort; neither is reachable through the Strands model
        # config yet, so it is declared rather than silently ignored.
        reason = (
            f"not accepted by {spec.id}"
            if not spec.supports_effort
            else "not yet forwarded through the harness layer"
        )
        notes.append(f"dropped effort: {reason}")

    if request.stream_reasoning:
        notes.append(
            "stream_reasoning is not yet forwarded; reasoning blocks are "
            "returned when the provider emits them"
        )

    return notes
