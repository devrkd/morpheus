"""Anthropic adapter — canonical types in, Messages API out.

Uses the official ``anthropic`` SDK. Three model-specific rules are enforced
here rather than left to the caller, because getting any of them wrong is a
400 rather than a degraded response:

* sampling parameters are dropped, because the SDK has no parameter for
  them since 1.x and the frontier models reject them anyway (see
  :attr:`~model_harness.core.registry.ModelSpec.supports_sampling`);
* reasoning depth goes through ``output_config.effort`` with adaptive
  thinking, never a token budget;
* an assistant turn this provider produced is replayed from its native
  payload, so thinking blocks (and their signatures) survive intact.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

import anthropic

from ..core.registry import ThinkingStyle
from ..core.types import (
    ImageBlock,
    Message,
    ReasoningBlock,
    Role,
    StopReason,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from ..errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from .base import (
    ProviderCall,
    ProviderResult,
    StreamSink,
    resolve_effort,
    sampling_params,
)
from .credentials import CredentialInfo, detect_anthropic

PROVIDER_NAME = "anthropic"

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "tool_use": StopReason.TOOL_USE,
    "refusal": StopReason.REFUSAL,
    # No server tools are declared, so a pause is not resumable from here;
    # report it as a completed turn rather than inventing a stop reason.
    "pause_turn": StopReason.END_TURN,
}


class AnthropicProvider:
    name = PROVIDER_NAME

    def __init__(
        self,
        *,
        api_key: str | None = None,
        profile: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.credential: CredentialInfo = detect_anthropic(
            configured_api_key=api_key, configured_profile=profile
        )

        # Only pass what was explicitly configured. Every omitted argument
        # leaves the SDK free to walk its own credential chain — env key, auth
        # token, `ant auth login` profile, workload identity federation,
        # default profile on disk. Passing `api_key=None` explicitly would not
        # break that, but passing an empty string would shadow the whole chain,
        # so falsy values are dropped rather than forwarded.
        kwargs: dict[str, Any] = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if profile:
            kwargs["profile"] = profile
        if base_url:
            kwargs["base_url"] = base_url

        self._client = anthropic.AsyncAnthropic(**kwargs)

    def available(self) -> bool:
        """Always true: the client is constructed regardless.

        A credential this class could not detect may still resolve inside the
        SDK, so refusing the request here would reject work that would have
        succeeded. An unauthenticated call fails as a provider auth error
        instead, which names the real problem.
        """
        return True

    def _require_client(self) -> anthropic.AsyncAnthropic:
        return self._client

    # --- request translation ---------------------------------------------

    def _build_kwargs(self, call: ProviderCall) -> tuple[dict[str, Any], list[str]]:
        spec = call.spec
        adjustments: list[str] = []

        native_messages: list[dict[str, Any]] = []
        for message in call.messages:
            native_messages.extend(self._to_native_messages(message))

        kwargs: dict[str, Any] = {
            "model": spec.native_id,
            "max_tokens": call.max_tokens,
            "messages": native_messages,
        }

        if call.tools:
            # Tool order is kept stable by the registry, which matters for
            # caching: `tools` renders before `system` and `messages`, so a
            # reordered tool list invalidates the entire cached prefix.
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in call.tools
            ]

        if call.system:
            # A list rather than a bare string so the prompt prefix can carry a
            # cache breakpoint; the system prompt is the most stable part of a
            # session and therefore the most valuable thing to cache.
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": block.text,
                    "cache_control": {"type": "ephemeral"},
                }
                for block in call.system
            ]

        # Auto-caches the last cacheable block, which for a session is the
        # accumulated transcript prefix. A prefix below the model's minimum
        # cacheable size simply won't cache — harmless, so this is
        # unconditional rather than gated on a token estimate.
        kwargs["cache_control"] = {"type": "ephemeral"}

        sampling, sampling_notes = sampling_params(call.inference_config, spec)
        kwargs.update(sampling)
        adjustments.extend(sampling_notes)

        if call.inference_config.stop_sequences and spec.supports_stop_sequences:
            kwargs["stop_sequences"] = call.inference_config.stop_sequences

        effort, effort_notes = resolve_effort(call.effort, spec)
        adjustments.extend(effort_notes)
        if effort is not None:
            kwargs["output_config"] = {"effort": effort.value}

        if spec.thinking is ThinkingStyle.ADAPTIVE:
            # Adaptive thinking, never a token budget: `budget_tokens` is
            # rejected outright by the current frontier models. `display`
            # defaults to omitted, so ask for a summary only when the caller
            # actually wants the reasoning back.
            thinking: dict[str, Any] = {"type": "adaptive"}
            if call.want_reasoning:
                thinking["display"] = "summarized"
            kwargs["thinking"] = thinking
        elif call.want_reasoning:
            adjustments.append(
                f"dropped stream_reasoning: {spec.id} has no server-side thinking control"
            )

        return kwargs, adjustments

    def _to_native_messages(self, message: Message) -> list[dict[str, Any]]:
        """Translate one canonical message.

        Returns a list to match the OpenAI adapter's signature, though
        Anthropic always yields exactly one message: it carries tool results
        as blocks on a user message rather than as separate messages.
        """
        if message.role is Role.ASSISTANT and message.provider == self.name and message.native:
            # Same-provider replay: hand back exactly what the API produced.
            # This is what keeps thinking blocks and their signatures valid.
            return [{"role": "assistant", "content": message.native}]

        content: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text:
                    content.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                content.append(
                    {"type": "image", "source": block.source.model_dump(exclude_none=True)}
                )
            elif isinstance(block, ToolUseBlock):
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif isinstance(block, ToolResultBlock):
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
            elif isinstance(block, ReasoningBlock):
                # Reasoning from another provider (or from a model that is no
                # longer serving this session) is not replayable — the target
                # would reject or ignore it. The text turn carries the content.
                continue

        if not content:
            content.append({"type": "text", "text": ""})
        return [{"role": message.role.value, "content": content}]

    # --- response translation --------------------------------------------

    def _to_canonical(self, response: Any) -> Message:
        blocks: list[TextBlock | ReasoningBlock | ToolUseBlock] = []
        for block in response.content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "thinking":
                blocks.append(ReasoningBlock(text=block.thinking or ""))
            elif block.type == "tool_use":
                blocks.append(
                    ToolUseBlock(id=block.id, name=block.name, input=dict(block.input or {}))
                )

        if not blocks:
            blocks.append(TextBlock(text=""))

        return Message(
            role=Role.ASSISTANT,
            content=blocks,
            provider=self.name,
            native=[b.model_dump(mode="json", exclude_none=True) for b in response.content],
        )

    @staticmethod
    def _to_usage(usage: Any) -> Usage:
        return Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    @staticmethod
    def _stop_details(response: Any) -> dict[str, Any] | None:
        # `stop_details` is populated only for a refusal; it is None for every
        # other stop reason, so guard before reading it.
        details = getattr(response, "stop_details", None)
        if details is None:
            return None
        return {
            "type": getattr(details, "type", None),
            "category": getattr(details, "category", None),
            "explanation": getattr(details, "explanation", None),
        }

    # --- entrypoints ------------------------------------------------------

    async def converse(self, call: ProviderCall) -> ProviderResult:
        client = self._require_client()
        kwargs, adjustments = self._build_kwargs(call)

        with _translate_errors(self.name, self.credential):
            response = await client.messages.create(**kwargs)

        return ProviderResult(
            message=self._to_canonical(response),
            stop_reason=_STOP_REASONS.get(response.stop_reason or "", StopReason.END_TURN),
            stop_details=self._stop_details(response),
            usage=self._to_usage(response.usage),
            native_model=response.model,
            adjustments=adjustments,
        )

    async def stream(self, call: ProviderCall, sink: StreamSink) -> AsyncIterator[StreamEvent]:
        client = self._require_client()
        kwargs, adjustments = self._build_kwargs(call)

        with _translate_errors(self.name, self.credential):
            async with client.messages.stream(**kwargs) as stream:
                yield StreamEvent(
                    type="message_start",
                    provider=self.name,
                    model=call.spec.id,
                    adjustments=adjustments,
                )

                open_blocks: dict[int, str] = {}
                async for event in stream:
                    if event.type == "content_block_start":
                        kind = event.content_block.type
                        if kind not in ("text", "thinking"):
                            continue
                        block_type = "reasoning" if kind == "thinking" else "text"
                        open_blocks[event.index] = block_type
                        yield StreamEvent(
                            type="content_block_start",
                            index=event.index,
                            block_type=block_type,  # type: ignore[arg-type]
                        )

                    elif event.type == "content_block_delta":
                        if event.index not in open_blocks:
                            continue
                        if event.delta.type == "text_delta":
                            text = event.delta.text
                        elif event.delta.type == "thinking_delta":
                            text = event.delta.thinking
                        else:
                            continue
                        yield StreamEvent(type="content_delta", index=event.index, text=text)

                    elif event.type == "content_block_stop":
                        if open_blocks.pop(event.index, None) is not None:
                            yield StreamEvent(type="content_block_stop", index=event.index)

                final = await stream.get_final_message()

        stop_reason = _STOP_REASONS.get(final.stop_reason or "", StopReason.END_TURN)
        usage = self._to_usage(final.usage)

        # Written before the terminal event so a consumer that stops reading
        # after `message_stop` still leaves the service a persistable turn.
        sink.result = ProviderResult(
            message=self._to_canonical(final),
            stop_reason=stop_reason,
            stop_details=self._stop_details(final),
            usage=usage,
            native_model=final.model,
            adjustments=adjustments,
        )

        # Emitted after the stream context closes so the terminal event always
        # reflects the fully accumulated message, including usage.
        yield StreamEvent(
            type="message_stop",
            provider=self.name,
            model=call.spec.id,
            native_model=final.model,
            stop_reason=stop_reason,
            usage=usage,
        )


class _translate_errors:
    """Map the SDK's typed exceptions onto the harness error vocabulary."""

    def __init__(self, provider: str, credential: CredentialInfo) -> None:
        self.provider = provider
        self.credential = credential

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is None:
            return False

        p = self.provider
        detail = f"{type(exc).__name__}: {exc}"

        if isinstance(exc, TypeError) and not self.credential.detected:
            # The SDK defers credential resolution to request time and signals
            # total failure with a bare TypeError rather than an APIError. Only
            # reinterpreted when detection already found nothing, so a genuine
            # TypeError in this adapter still propagates as a bug.
            raise ProviderUnavailableError(
                "No Anthropic credential could be resolved. Set ANTHROPIC_API_KEY, "
                "run `ant auth login`, or configure workload identity federation.",
                provider=p,
                detail=detail,
            )

        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            raise ProviderAuthError(
                "Anthropic rejected the harness credential", provider=p, detail=detail
            )
        if isinstance(exc, anthropic.RateLimitError):
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                raw = response.headers.get("retry-after")
                retry_after = int(raw) if raw and raw.isdigit() else None
            raise ProviderRateLimitError(
                "Anthropic rate limit reached",
                provider=p,
                detail=detail,
                retry_after=retry_after,
            )
        if isinstance(exc, anthropic.BadRequestError | anthropic.NotFoundError):
            raise ProviderBadRequestError(
                "Anthropic rejected the request", provider=p, detail=detail
            )
        if isinstance(exc, anthropic.APITimeoutError):
            raise ProviderTimeoutError("Anthropic request timed out", provider=p, detail=detail)
        if isinstance(exc, anthropic.APIStatusError):
            if exc.status_code >= 500:
                raise ProviderServerError("Anthropic server error", provider=p, detail=detail)
            raise ProviderBadRequestError("Anthropic error", provider=p, detail=detail)
        if isinstance(exc, anthropic.APIConnectionError):
            raise ProviderConnectionError("Could not reach Anthropic", provider=p, detail=detail)
        return False
