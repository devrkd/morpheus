"""OpenAI adapter — canonical types in, Chat Completions out.

Uses the official ``openai`` SDK against ``chat.completions``. Two deliberate
limitations, both reported back to the client as ``adjustments`` rather than
hidden:

* ``top_k`` has no equivalent and is dropped;
* reasoning summaries are not returned by this endpoint, so a request with
  ``stream_reasoning`` on an OpenAI model gets text only. Reasoning tokens are
  still billed and are reported in ``usage.reasoning_tokens``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

import openai

from ..core.registry import ThinkingStyle
from ..core.types import (
    Base64ImageSource,
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
from .credentials import CredentialInfo, CredentialSource, detect_openai

PROVIDER_NAME = "openai"

_FINISH_REASONS: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "content_filter": StopReason.REFUSAL,
}


class OpenAIProvider:
    name = PROVIDER_NAME

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.credential: CredentialInfo = detect_openai(configured_api_key=api_key)

        # As with the Anthropic adapter, omitted arguments let the SDK resolve
        # its own credential (env key or workload identity).
        kwargs: dict[str, Any] = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if organization:
            kwargs["organization"] = organization
        if project:
            kwargs["project"] = project
        if base_url:
            kwargs["base_url"] = base_url

        self._client: openai.AsyncOpenAI | None
        try:
            self._client = openai.AsyncOpenAI(**kwargs)
        except openai.OpenAIError:
            # Unlike the Anthropic SDK, this one raises at construction when it
            # finds no credential at all. That makes the failure authoritative,
            # so the adapter can honestly report itself unavailable rather than
            # deferring to a doomed request.
            self._client = None
            self.credential = CredentialInfo(
                CredentialSource.UNDETECTED,
                "the OpenAI SDK found no usable credential at startup",
            )

    def available(self) -> bool:
        return self._client is not None

    def _require_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            raise ProviderUnavailableError(
                "No OpenAI credential is configured, so OpenAI models cannot be "
                "served. Set OPENAI_API_KEY or configure workload identity.",
                provider=self.name,
            )
        return self._client

    # --- request translation ---------------------------------------------

    def _build_kwargs(self, call: ProviderCall) -> tuple[dict[str, Any], list[str]]:
        spec = call.spec
        adjustments: list[str] = []
        is_reasoning = spec.thinking is ThinkingStyle.REASONING_EFFORT

        messages: list[dict[str, Any]] = []
        if call.system:
            messages.append(
                {
                    "role": "system",
                    "content": "\n\n".join(block.text for block in call.system),
                }
            )
        for message in call.messages:
            messages.extend(self._to_native_messages(message))

        kwargs: dict[str, Any] = {"model": spec.native_id, "messages": messages}

        if call.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in call.tools
            ]

        # Reasoning models count reasoning tokens against the output ceiling
        # and take `max_completion_tokens`; the older models take `max_tokens`.
        if is_reasoning:
            kwargs["max_completion_tokens"] = call.max_tokens
        else:
            kwargs["max_tokens"] = call.max_tokens

        sampling, sampling_notes = sampling_params(call.inference_config, spec)
        adjustments.extend(sampling_notes)
        if "top_k" in sampling:
            del sampling["top_k"]
            adjustments.append("dropped top_k: no equivalent in the OpenAI API")
        kwargs.update(sampling)

        if call.inference_config.stop_sequences and spec.supports_stop_sequences:
            kwargs["stop"] = call.inference_config.stop_sequences

        effort, effort_notes = resolve_effort(call.effort, spec)
        adjustments.extend(effort_notes)
        if effort is not None and is_reasoning:
            kwargs["reasoning_effort"] = effort.value

        if call.want_reasoning:
            adjustments.append(
                "stream_reasoning has no effect: the OpenAI chat completions API "
                "does not return reasoning summaries"
            )

        return kwargs, adjustments

    def _to_native_messages(self, message: Message) -> list[dict[str, Any]]:
        """Translate one canonical message into one *or more* native messages.

        This is where the two vendors diverge most. Anthropic carries tool
        results as blocks on a user message; OpenAI wants a separate
        ``tool``-role message per result. Keeping the canonical form
        Anthropic-shaped and fanning out here is what lets a transcript full
        of tool calls replay against either provider.
        """
        if message.role is Role.ASSISTANT and message.provider == self.name and message.native:
            return [message.native]

        if message.role is Role.ASSISTANT:
            # Assistant turns go back as text plus any tool calls. A reasoning
            # block from this or another provider is not replayable here.
            tool_uses = [b for b in message.content if isinstance(b, ToolUseBlock)]
            native: dict[str, Any] = {"role": "assistant", "content": message.text() or None}
            if tool_uses:
                native["tool_calls"] = [
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                    for block in tool_uses
                ]
            return [native]

        # A user turn may carry tool results, which become their own messages.
        # They must precede any new user text, because each one answers a tool
        # call from the preceding assistant turn.
        results = [b for b in message.content if isinstance(b, ToolResultBlock)]
        out: list[dict[str, Any]] = [
            {
                "role": "tool",
                "tool_call_id": block.tool_use_id,
                "content": block.content or ("error" if block.is_error else ""),
            }
            for block in results
        ]

        parts: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text:
                    parts.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                source = block.source
                url = (
                    f"data:{source.media_type};base64,{source.data}"
                    if isinstance(source, Base64ImageSource)
                    else source.url
                )
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif isinstance(block, ReasoningBlock | ToolResultBlock):
                continue

        if parts:
            out.append({"role": message.role.value, "content": parts})
        elif not out:
            out.append({"role": message.role.value, "content": [{"type": "text", "text": ""}]})
        return out

    # --- response translation --------------------------------------------

    def _to_canonical(self, choice: Any) -> Message:
        native_message = choice.message
        text = native_message.content or ""
        refusal = getattr(native_message, "refusal", None)
        tool_calls = getattr(native_message, "tool_calls", None) or []

        blocks: list[TextBlock | ReasoningBlock | ToolUseBlock] = [TextBlock(text=text)]
        native: dict[str, Any] = {"role": "assistant", "content": text or None}

        if tool_calls:
            native["tool_calls"] = []
            for tc in tool_calls:
                raw = tc.function.arguments or "{}"
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    # Malformed arguments are handed to the tool layer as an
                    # empty input rather than crashing the turn; the tool
                    # reports the schema violation back to the model.
                    parsed = {}
                blocks.append(
                    ToolUseBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=parsed if isinstance(parsed, dict) else {},
                    )
                )
                native["tool_calls"].append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": raw},
                    }
                )

        if refusal:
            # Surface the refusal as the turn's text so the transcript stays
            # readable, but keep it out of the replayed payload — the API
            # rejects an assistant message carrying both.
            blocks = [TextBlock(text=refusal)]
            native = {"role": "assistant", "content": refusal}

        return Message(
            role=Role.ASSISTANT,
            content=blocks,
            provider=self.name,
            native=native,
        )

    @staticmethod
    def _to_usage(usage: Any) -> Usage:
        if usage is None:
            return Usage()
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        return Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_read_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        )

    @staticmethod
    def _stop_reason(choice: Any) -> StopReason:
        if getattr(choice.message, "refusal", None):
            return StopReason.REFUSAL
        return _FINISH_REASONS.get(choice.finish_reason or "", StopReason.END_TURN)

    # --- entrypoints ------------------------------------------------------

    async def converse(self, call: ProviderCall) -> ProviderResult:
        client = self._require_client()
        kwargs, adjustments = self._build_kwargs(call)

        with _translate_errors(self.name):
            response = await client.chat.completions.create(**kwargs)

        if not response.choices:
            raise ProviderServerError(
                "OpenAI returned no choices for the request", provider=self.name
            )
        choice = response.choices[0]

        return ProviderResult(
            message=self._to_canonical(choice),
            stop_reason=self._stop_reason(choice),
            usage=self._to_usage(response.usage),
            native_model=response.model,
            adjustments=adjustments,
        )

    async def stream(self, call: ProviderCall, sink: StreamSink) -> AsyncIterator[StreamEvent]:
        client = self._require_client()
        kwargs, adjustments = self._build_kwargs(call)
        # Usage is omitted from a stream unless explicitly requested, and the
        # service needs it to report the turn's cost.
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        parts: list[str] = []
        finish_reason: str | None = None
        refusal_parts: list[str] = []
        usage: Usage = Usage()
        native_model = call.spec.native_id
        started = False

        with _translate_errors(self.name):
            stream = await client.chat.completions.create(**kwargs)

            yield StreamEvent(
                type="message_start",
                provider=self.name,
                model=call.spec.id,
                adjustments=adjustments,
            )

            async for chunk in stream:
                if chunk.model:
                    native_model = chunk.model
                if chunk.usage is not None:
                    usage = self._to_usage(chunk.usage)
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue

                refusal = getattr(delta, "refusal", None)
                if refusal:
                    refusal_parts.append(refusal)

                text = delta.content
                if not text:
                    continue

                if not started:
                    started = True
                    yield StreamEvent(type="content_block_start", index=0, block_type="text")
                parts.append(text)
                yield StreamEvent(type="content_delta", index=0, text=text)

        if started:
            yield StreamEvent(type="content_block_stop", index=0)

        refusal_text = "".join(refusal_parts)
        final_text = refusal_text or "".join(parts)
        stop_reason = (
            StopReason.REFUSAL
            if refusal_text
            else _FINISH_REASONS.get(finish_reason or "", StopReason.END_TURN)
        )

        sink.result = ProviderResult(
            message=Message(
                role=Role.ASSISTANT,
                content=[TextBlock(text=final_text)],
                provider=self.name,
                native={"role": "assistant", "content": final_text},
            ),
            stop_reason=stop_reason,
            usage=usage,
            native_model=native_model,
            adjustments=adjustments,
        )

        yield StreamEvent(
            type="message_stop",
            provider=self.name,
            model=call.spec.id,
            native_model=native_model,
            stop_reason=stop_reason,
            usage=usage,
        )


class _translate_errors:
    """Map the SDK's typed exceptions onto the harness error vocabulary."""

    def __init__(self, provider: str) -> None:
        self.provider = provider

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

        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            raise ProviderAuthError(
                "OpenAI rejected the harness credential", provider=p, detail=detail
            )
        if isinstance(exc, openai.RateLimitError):
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                raw = response.headers.get("retry-after")
                retry_after = int(raw) if raw and raw.isdigit() else None
            raise ProviderRateLimitError(
                "OpenAI rate limit reached",
                provider=p,
                detail=detail,
                retry_after=retry_after,
            )
        if isinstance(exc, openai.BadRequestError | openai.NotFoundError):
            raise ProviderBadRequestError("OpenAI rejected the request", provider=p, detail=detail)
        if isinstance(exc, openai.APITimeoutError):
            raise ProviderTimeoutError("OpenAI request timed out", provider=p, detail=detail)
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code >= 500:
                raise ProviderServerError("OpenAI server error", provider=p, detail=detail)
            raise ProviderBadRequestError("OpenAI error", provider=p, detail=detail)
        if isinstance(exc, openai.APIConnectionError):
            raise ProviderConnectionError("Could not reach OpenAI", provider=p, detail=detail)
        return False
