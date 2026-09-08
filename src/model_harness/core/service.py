"""The orchestration layer: routing, session context, and provider dispatch.

This is where "the context remains available" is actually implemented. A
session owns one canonical transcript. On every turn the service replays that
whole transcript to whichever provider the requested model resolves to, so a
client can switch from ``claude-opus-5`` to ``gpt-5`` mid-conversation and the
new model still sees everything said so far.

What survives a switch, precisely:

* user turns — fully, including images;
* assistant text — fully;
* provider-native assistant state (Anthropic thinking blocks and their
  signatures) — only while the same provider keeps serving the session. On a
  switch those blocks are left out of the replay, because the other vendor
  would reject or ignore them. The switch is reported in ``adjustments`` rather
  than passing silently.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from ..auth.principals import Principal
from ..config import Settings
from ..errors import (
    InvalidRequestError,
    ModelNotPermittedError,
    ProviderUnavailableError,
    SessionNotFoundError,
    ToolNotPermittedError,
    UnknownModelError,
)
from ..providers.base import LLMProvider, ProviderCall, ProviderResult, StreamSink
from ..sessions.base import Session, SessionStore
from ..tools.base import Tool, ToolRegistry
from ..tools.builtin import execute as execute_tool
from .registry import ModelSpec, Provider, resolve
from .types import (
    ConverseOutput,
    ConverseRequest,
    ConverseResponse,
    Message,
    ReasoningBlock,
    Role,
    StopReason,
    StreamEvent,
    SystemBlock,
    ToolInvocation,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


def _add_usage(total: Usage, delta: Usage) -> Usage:
    """Sum usage across the loop's iterations.

    A tool-using turn costs every request in the loop, not just the last one,
    so reporting only the final response's usage would understate it badly.
    """
    return Usage(
        input_tokens=total.input_tokens + delta.input_tokens,
        output_tokens=total.output_tokens + delta.output_tokens,
        cache_read_tokens=total.cache_read_tokens + delta.cache_read_tokens,
        cache_write_tokens=total.cache_write_tokens + delta.cache_write_tokens,
        reasoning_tokens=total.reasoning_tokens + delta.reasoning_tokens,
    )


class ConverseService:
    DEFAULT_TOOL_ITERATIONS = 5
    """Round trips allowed before the loop gives up.

    An uncapped loop is a runaway bill: every iteration is a full request
    carrying the whole transcript plus each tool result so far.
    """

    def __init__(
        self,
        *,
        providers: dict[str, LLMProvider],
        store: SessionStore,
        settings: Settings,
        tools: ToolRegistry | None = None,
    ) -> None:
        self._providers = providers
        self._store = store
        self._settings = settings
        self._tools = tools or ToolRegistry()

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    # --- public API -------------------------------------------------------

    async def converse(self, request: ConverseRequest, principal: Principal) -> ConverseResponse:
        spec, provider = self._route(request.model, principal)
        session = await self._load_session(request, principal)
        tools, tool_notes = self._resolve_tools(request, principal)
        call, adjustments = self._prepare(
            request, session, spec, provider, principal, streaming=False, tools=tools
        )
        adjustments.extend(tool_notes)

        by_name = {tool.name: tool for tool in tools}
        max_iterations = request.max_tool_iterations or self.DEFAULT_TOOL_ITERATIONS

        # Everything to append to the session on success: the caller's turn,
        # then each assistant tool call and its results that the loop produced.
        pending: list[Message] = list(request.messages)
        invocations: list[ToolInvocation] = []
        usage = Usage()
        iterations = 0

        started = time.perf_counter()
        try:
            while True:
                iterations += 1
                result = await provider.converse(call)
                usage = _add_usage(usage, result.usage)

                tool_uses = [b for b in result.message.content if isinstance(b, ToolUseBlock)]
                if result.stop_reason is not StopReason.TOOL_USE or not tool_uses:
                    break

                if iterations >= max_iterations:
                    # Stop, but keep the transcript valid. Both providers reject
                    # a tool_use with no matching result on the next request, so
                    # a dangling call would break every later turn in this
                    # session. Synthetic results record what happened.
                    pending.append(result.message)
                    pending.append(
                        Message(
                            role=Role.USER,
                            content=[
                                ToolResultBlock(
                                    tool_use_id=block.id,
                                    name=block.name,
                                    content=(
                                        "Not executed: this turn reached its limit of "
                                        f"{max_iterations} tool iterations."
                                    ),
                                    is_error=True,
                                )
                                for block in tool_uses
                            ],
                        )
                    )
                    adjustments.append(
                        f"stopped after {max_iterations} tool iterations without a "
                        "final answer; raise max_tool_iterations to allow more"
                    )
                    break

                pending.append(result.message)
                result_blocks: list[ToolResultBlock] = []

                for block in tool_uses:
                    tool = by_name.get(block.name)
                    if tool is None:
                        # The model invented a tool, or named one it was not
                        # granted. Returned as an error result so it can correct
                        # itself, rather than failing the whole turn.
                        message = (
                            f"Error: tool '{block.name}' is not available. "
                            f"Available tools: {sorted(by_name) or 'none'}."
                        )
                        output, is_error, duration = message, True, 0
                    else:
                        output, is_error, duration = await execute_tool(tool, block.input)

                    invocations.append(
                        ToolInvocation(
                            name=block.name,
                            tool_use_id=block.id,
                            input=block.input,
                            output=output,
                            is_error=is_error,
                            duration_ms=duration,
                        )
                    )
                    result_blocks.append(
                        ToolResultBlock(
                            tool_use_id=block.id,
                            name=block.name,
                            content=output,
                            is_error=is_error,
                        )
                    )

                pending.append(Message(role=Role.USER, content=result_blocks))
                call.messages = [*session.messages, *pending]
        except Exception:
            await self._discard_if_empty(session)
            raise

        latency_ms = int((time.perf_counter() - started) * 1000)
        await self._commit(session, pending, result, provider.name, spec, system=request.system)

        return ConverseResponse(
            session_id=session.id,
            provider=provider.name,
            model=spec.id,
            native_model=result.native_model,
            output=ConverseOutput(message=result.message),
            stop_reason=result.stop_reason,
            stop_details=result.stop_details,
            usage=usage,
            latency_ms=latency_ms,
            tool_calls=invocations,
            iterations=iterations,
            adjustments=[*adjustments, *result.adjustments],
        )

    async def converse_stream(
        self, request: ConverseRequest, principal: Principal
    ) -> tuple[str, AsyncIterator[StreamEvent]]:
        """Return the session id and the canonical event stream.

        The id is returned up front so the route can put it in a response
        header before the first byte of the body is written — a client
        starting a new session should not have to wait for the stream to
        finish to learn where to send its next turn.
        """
        spec, provider = self._route(request.model, principal)
        session = await self._load_session(request, principal)
        if request.tools:
            # A tool loop makes several provider round trips, so one response
            # stream cannot represent it honestly. Rejecting beats silently
            # ignoring the tools the caller asked for.
            raise InvalidRequestError(
                "Tools are not supported on /v1/converse-stream yet. Use "
                "/v1/converse for a turn that needs tools."
            )

        call, adjustments = self._prepare(
            request, session, spec, provider, principal, streaming=True
        )

        async def generate() -> AsyncIterator[StreamEvent]:
            sink = StreamSink()
            pending = list(adjustments)
            try:
                async for event in provider.stream(call, sink):
                    if event.type == "message_start":
                        event.session_id = session.id
                        event.adjustments = [*pending, *(event.adjustments or [])]
                    elif event.type == "message_stop":
                        event.session_id = session.id
                    yield event
            finally:
                # Persist whatever completed, even if the client disconnected
                # mid-stream: a partial assistant turn is still context the
                # next turn needs, and dropping it would silently desync the
                # session from what the user saw.
                if sink.result is not None:
                    await self._commit(
                        session,
                        list(request.messages),
                        sink.result,
                        provider.name,
                        spec,
                        system=request.system,
                    )
                else:
                    await self._discard_if_empty(session)

        return session.id, generate()

    async def get_session(self, session_id: str, principal: Principal) -> Session | None:
        session = await self._store.get(session_id)
        if session is None or session.owner != principal.id:
            # A session owned by someone else reads as absent rather than
            # forbidden. A 403 would confirm the id exists, which is exactly
            # what someone probing for other callers' sessions wants to learn.
            return None
        return session

    async def delete_session(self, session_id: str, principal: Principal) -> bool:
        if await self.get_session(session_id, principal) is None:
            return False
        return await self._store.delete(session_id)

    async def list_sessions(self, principal: Principal, limit: int = 100) -> list[str]:
        return await self._store.list_ids(limit, owner=principal.id)

    async def list_session_summaries(
        self, principal: Principal, limit: int = 100
    ) -> list[dict[str, object]]:
        """Metadata for each of the principal's sessions, newest first.

        Exists so a UI can render a session list in one request instead of
        fanning out over the id list.
        """
        summaries = []
        for session_id in await self._store.list_ids(limit, owner=principal.id):
            session = await self._store.get(session_id)
            if session is not None and session.owner == principal.id:
                summaries.append(session.summary())
        return summaries

    def provider_status(self) -> dict[str, dict[str, object]]:
        status: dict[str, dict[str, object]] = {}
        for name, provider in self._providers.items():
            credential = getattr(provider, "credential", None)
            status[name] = {
                "available": provider.available(),
                "credential_source": credential.source.value if credential else "unknown",
                "credential_detected": bool(credential and credential.detected),
                "note": credential.note if credential else "",
            }
        return status

    # --- internals --------------------------------------------------------

    def _route(self, model_id: str, principal: Principal) -> tuple[ModelSpec, LLMProvider]:
        spec = resolve(model_id)
        if spec is None:
            raise UnknownModelError(
                f"Unknown model '{model_id}'. See GET /v1/models for the catalog."
            )

        if not principal.may_use(spec.id):
            # Checked against the canonical id, not the requested string, so an
            # alias cannot be used to slip past an allowlist.
            raise ModelNotPermittedError(
                f"Principal '{principal.id}' is not permitted to use model '{spec.id}'"
            )

        provider = self._providers.get(spec.provider.value)
        if provider is None:
            raise ProviderUnavailableError(
                f"No adapter registered for provider '{spec.provider}'",
                provider=spec.provider.value,
            )
        if not provider.available():
            raise ProviderUnavailableError(
                f"Provider '{spec.provider}' has no credential configured, so "
                f"model '{spec.id}' cannot be served",
                provider=spec.provider.value,
            )
        return spec, provider

    async def _load_session(self, request: ConverseRequest, principal: Principal) -> Session:
        if request.session_id is None:
            return await self._store.create(owner=principal.id)

        session = await self._store.get(request.session_id)
        if session is not None:
            if session.owner != principal.id:
                # Not adopted, unlike an unknown id: writing into a live
                # session belonging to another principal would leak this turn
                # to them and their history to this caller.
                raise SessionNotFoundError(f"No session '{request.session_id}'")
            return session

        # An unknown or expired id becomes a new session under that id rather
        # than a 404. A client holding a stale id can keep going; a hard error
        # here would mean losing the turn the user just typed.
        return await self._store.create(request.session_id, owner=principal.id)

    def _prepare(
        self,
        request: ConverseRequest,
        session: Session,
        spec: ModelSpec,
        provider: LLMProvider,
        principal: Principal,
        *,
        streaming: bool,
        tools: list[Tool] | None = None,
    ) -> tuple[ProviderCall, list[str]]:
        self._validate_incoming(request.messages)

        adjustments: list[str] = []

        system = request.system if request.system is not None else session.system
        if (
            request.system is not None
            and session.system is not None
            and request.system != session.system
            and session.messages
        ):
            adjustments.append(
                "system prompt changed mid-session: the provider's cached prefix "
                "for this session is invalidated"
            )

        if session.last_provider and session.last_provider != provider.name:
            adjustments.append(
                f"provider switched {session.last_provider} -> {provider.name}: prior "
                "assistant turns are replayed as text, so provider-native reasoning "
                "state from earlier turns is not carried over"
            )

        transcript = [*session.messages, *request.messages]
        if transcript[0].role is not Role.USER:
            raise InvalidRequestError("The first message in a conversation must have role 'user'")

        default = (
            self._settings.default_max_tokens_streaming
            if streaming
            else self._settings.default_max_tokens
        )
        requested = request.inference_config.max_tokens or default
        max_tokens = min(requested, spec.max_output_tokens)
        if requested > spec.max_output_tokens:
            adjustments.append(
                f"max_tokens {requested} lowered to {spec.max_output_tokens}, "
                f"the output ceiling for {spec.id}"
            )

        if principal.max_tokens_per_turn is not None and (
            max_tokens > principal.max_tokens_per_turn
        ):
            adjustments.append(
                f"max_tokens {max_tokens} lowered to {principal.max_tokens_per_turn}, "
                f"the per-turn ceiling for principal '{principal.id}'"
            )
            max_tokens = principal.max_tokens_per_turn

        call = ProviderCall(
            spec=spec,
            system=system,
            messages=transcript,
            inference_config=request.inference_config,
            effort=request.effort,
            want_reasoning=request.stream_reasoning,
            max_tokens=max_tokens,
            tools=tools or [],
        )
        return call, adjustments

    def _resolve_tools(
        self, request: ConverseRequest, principal: Principal
    ) -> tuple[list[Tool], list[str]]:
        """Resolve requested tool names, refusing anything not granted.

        Two gates, both required: the tool must exist, and the principal must
        be allowed it. A dangerous tool has to be granted explicitly — a
        principal with no ``allowed_tools`` gets the safe tools only, so
        side-effecting capability is never acquired by default.
        """
        if not request.tools:
            return [], []

        resolved: list[Tool] = []
        for name in request.tools:
            tool = self._tools.get(name)
            if tool is None:
                raise InvalidRequestError(
                    f"Unknown tool '{name}'. See GET /v1/tools for what is available."
                )
            if not principal.may_use_tool(tool.name, dangerous=tool.dangerous):
                extra = (
                    " (it can affect state outside this service, so it must be granted explicitly)"
                    if tool.dangerous
                    else ""
                )
                raise ToolNotPermittedError(
                    f"Principal '{principal.id}' is not permitted to use tool '{tool.name}'{extra}"
                )
            resolved.append(tool)

        # Registry order, so the rendered tool list is stable turn to turn.
        # `tools` renders before `system` and `messages`, so a reordered list
        # would invalidate the provider's whole cached prefix.
        order = {name: i for i, name in enumerate(self._tools.names())}
        resolved.sort(key=lambda t: order.get(t.name, 0))
        return resolved, []

    @staticmethod
    def _validate_incoming(messages: list[Message]) -> None:
        if messages[-1].role is not Role.USER:
            raise InvalidRequestError(
                "The last message of a request must have role 'user' — send only "
                "the new turn; the stored transcript is prepended for you"
            )
        for message in messages:
            if any(
                isinstance(block, ReasoningBlock | ToolUseBlock | ToolResultBlock)
                for block in message.content
            ):
                raise InvalidRequestError(
                    "Reasoning, tool_use and tool_result blocks are produced by the "
                    "service and cannot be supplied by a client"
                )

    async def _discard_if_empty(self, session: Session) -> None:
        """Drop a session that never recorded a turn.

        A session is allocated before the provider is called, so a failed
        first request would otherwise leave an empty one behind — visible
        clutter in any session list, and misleading, since nothing was ever
        said in it. Sessions that already hold turns are untouched.
        """
        if not session.messages:
            await self._store.delete(session.id)

    async def _commit(
        self,
        session: Session,
        turns: list[Message],
        result: ProviderResult,
        provider_name: str,
        spec: ModelSpec,
        *,
        system: list[SystemBlock] | None = None,
    ) -> None:
        """Append a completed turn.

        ``turns`` is everything before the final assistant message: the
        caller's message plus, for a tool loop, each intermediate assistant
        tool call and its results. They commit together, so a session never
        holds a tool call without its matching result.
        """
        if system is not None:
            session.system = system

        session.messages.extend(turns)
        session.messages.append(result.message)
        session.last_provider = provider_name
        session.last_model = spec.id
        session.turn_count += 1
        await self._store.save(session)


def build_providers(settings: Settings) -> dict[str, LLMProvider]:
    """Instantiate every known adapter.

    Each adapter resolves its own credential through its SDK's chain and
    reports what it found; see ``providers/credentials.py``. An adapter that
    could not detect one is still registered, so a request either succeeds or
    fails with an error naming the configuration problem.
    """
    from ..providers.anthropic_provider import AnthropicProvider
    from ..providers.openai_provider import OpenAIProvider

    return {
        Provider.ANTHROPIC.value: AnthropicProvider(
            api_key=settings.anthropic_api_key,
            profile=settings.anthropic_profile,
            base_url=settings.anthropic_base_url,
            timeout=settings.request_timeout_seconds,
        ),
        Provider.OPENAI.value: OpenAIProvider(
            api_key=settings.openai_api_key,
            organization=settings.openai_organization,
            project=settings.openai_project,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
        ),
    }
