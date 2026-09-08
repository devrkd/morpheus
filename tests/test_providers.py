"""Adapter translation, exercised without any network calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic
import pytest

from model_harness.core.registry import resolve
from model_harness.core.types import (
    Base64ImageSource,
    Effort,
    ImageBlock,
    InferenceConfig,
    Message,
    ReasoningBlock,
    Role,
    StopReason,
    SystemBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UrlImageSource,
)
from model_harness.errors import ProviderUnavailableError
from model_harness.providers.anthropic_provider import AnthropicProvider
from model_harness.providers.base import ProviderCall
from model_harness.providers.credentials import CredentialInfo, CredentialSource
from model_harness.providers.openai_provider import OpenAIProvider
from model_harness.tools.base import Tool as ToolDef


def call(model: str, messages: list[Message], **kw) -> ProviderCall:
    return ProviderCall(
        spec=resolve(model),
        system=kw.pop("system", None),
        messages=messages,
        inference_config=kw.pop("inference_config", InferenceConfig()),
        effort=kw.pop("effort", None),
        want_reasoning=kw.pop("want_reasoning", False),
        max_tokens=kw.pop("max_tokens", 1024),
        tools=kw.pop("tools", []),
    )


def user(text: str) -> Message:
    return Message(role=Role.USER, content=[TextBlock(text=text)])


# --- shared -------------------------------------------------------------


def test_the_two_adapters_report_availability_differently_on_purpose():
    """The asymmetry follows the SDKs, and is deliberate.

    The OpenAI SDK raises at construction when it finds no credential, so that
    answer is authoritative and the adapter reports itself unavailable. The
    Anthropic SDK constructs regardless and resolves lazily, so its adapter
    stays available and lets the request decide — refusing up front would
    reject work that a profile or workload identity would have authenticated.
    """
    assert OpenAIProvider(api_key=None).available() is False
    assert AnthropicProvider(api_key="explicit-key").available() is True


async def test_an_unconfigured_openai_provider_raises_before_any_call():
    with pytest.raises(ProviderUnavailableError):
        await OpenAIProvider(api_key=None).converse(call("gpt-5", [user("hi")]))


async def test_an_unresolvable_anthropic_credential_becomes_unavailable_not_a_crash():
    """The SDK signals a total credential failure with a bare TypeError at
    request time; the adapter must translate that, not leak a 500."""
    provider = AnthropicProvider(api_key="placeholder")
    provider.credential = CredentialInfo(CredentialSource.UNDETECTED, "forced for this test")
    provider._client = anthropic.AsyncAnthropic(api_key=None, base_url="http://127.0.0.1:1")

    with pytest.raises(ProviderUnavailableError, match="No Anthropic credential"):
        await provider.converse(call("claude-opus-5", [user("hi")]))


# --- Anthropic ----------------------------------------------------------


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key")


def test_anthropic_sends_adaptive_thinking_and_effort(anthropic_provider):
    kwargs, adjustments = anthropic_provider._build_kwargs(
        call("claude-opus-5", [user("hi")], effort=Effort.MAX, want_reasoning=True)
    )
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["output_config"] == {"effort": "max"}
    assert "budget_tokens" not in kwargs
    assert adjustments == []


def test_anthropic_omits_reasoning_display_unless_asked(anthropic_provider):
    kwargs, _ = anthropic_provider._build_kwargs(call("claude-opus-5", [user("hi")]))
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_anthropic_drops_sampling_for_frontier_models(anthropic_provider):
    kwargs, adjustments = anthropic_provider._build_kwargs(
        call(
            "claude-opus-5",
            [user("hi")],
            inference_config=InferenceConfig(temperature=0.5, top_k=40),
        )
    )
    assert "temperature" not in kwargs and "top_k" not in kwargs
    assert any("dropped" in a for a in adjustments)


def test_anthropic_drops_sampling_on_older_models_too(anthropic_provider):
    """The SDK removed these parameters in 1.x, so no Anthropic model gets them."""
    kwargs, adjustments = anthropic_provider._build_kwargs(
        call(
            "claude-haiku-4-5",
            [user("hi")],
            inference_config=InferenceConfig(temperature=0.5, top_k=40),
        )
    )
    assert "temperature" not in kwargs and "top_k" not in kwargs
    assert any("dropped" in a for a in adjustments)
    # Haiku 4.5 predates adaptive thinking; the harness sends neither knob.
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


@pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"])
def test_every_anthropic_kwarg_is_accepted_by_the_installed_sdk(anthropic_provider, model):
    """Guards against SDK drift.

    `anthropic` 1.x dropped `temperature`/`top_p`/`top_k` from
    `messages.create` outright, so a parameter the harness invents or an
    upstream removal shows up here as a failure rather than as a TypeError in
    production.
    """
    import inspect

    from anthropic.resources.messages import AsyncMessages

    accepted = set(inspect.signature(AsyncMessages.create).parameters)
    stream_accepted = set(inspect.signature(AsyncMessages.stream).parameters)

    kwargs, _ = anthropic_provider._build_kwargs(
        call(
            model,
            [user("hi")],
            system=[SystemBlock(text="s")],
            effort=Effort.HIGH,
            want_reasoning=True,
            inference_config=InferenceConfig(
                temperature=0.5, top_p=0.9, top_k=40, stop_sequences=["STOP"]
            ),
        )
    )
    assert set(kwargs) <= accepted
    assert set(kwargs) <= stream_accepted


@pytest.mark.parametrize("model", ["gpt-5", "gpt-4o"])
def test_every_openai_kwarg_is_accepted_by_the_installed_sdk(openai_provider, model):
    import inspect

    from openai.resources.chat.completions import AsyncCompletions

    accepted = set(inspect.signature(AsyncCompletions.create).parameters)
    kwargs, _ = openai_provider._build_kwargs(
        call(
            model,
            [user("hi")],
            system=[SystemBlock(text="s")],
            effort=Effort.HIGH,
            inference_config=InferenceConfig(
                temperature=0.5, top_p=0.9, top_k=40, stop_sequences=["STOP"]
            ),
        )
    )
    assert set(kwargs) <= accepted


def test_anthropic_caches_the_system_prompt_and_prefix(anthropic_provider):
    kwargs, _ = anthropic_provider._build_kwargs(
        call("claude-opus-5", [user("hi")], system=[SystemBlock(text="be terse")])
    )
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["cache_control"] == {"type": "ephemeral"}


def test_anthropic_replays_its_own_native_assistant_turn(anthropic_provider):
    native = [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "answer"},
    ]
    history = [
        user("q"),
        Message(
            role=Role.ASSISTANT,
            content=[ReasoningBlock(text="hmm"), TextBlock(text="answer")],
            provider="anthropic",
            native=native,
        ),
        user("q2"),
    ]
    kwargs, _ = anthropic_provider._build_kwargs(call("claude-opus-5", history))
    assert kwargs["messages"][1] == {"role": "assistant", "content": native}


def test_anthropic_falls_back_to_text_for_a_foreign_assistant_turn(anthropic_provider):
    history = [
        user("q"),
        Message(
            role=Role.ASSISTANT,
            content=[ReasoningBlock(text="leaky"), TextBlock(text="answer")],
            provider="openai",
            native={"role": "assistant", "content": "answer"},
        ),
        user("q2"),
    ]
    kwargs, _ = anthropic_provider._build_kwargs(call("claude-opus-5", history))
    assert kwargs["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
    }


def test_anthropic_translates_images(anthropic_provider):
    message = Message(
        role=Role.USER,
        content=[
            ImageBlock(source=Base64ImageSource(media_type="image/png", data="AAA")),
            TextBlock(text="what is this"),
        ],
    )
    kwargs, _ = anthropic_provider._build_kwargs(call("claude-opus-5", [message]))
    assert kwargs["messages"][0]["content"][0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
    }


# --- Anthropic response translation --------------------------------------


@dataclass
class _Block:
    type: str
    text: str | None = None
    thinking: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 5
    cache_read_input_tokens: int = 3
    cache_creation_input_tokens: int = 2


@dataclass
class _Response:
    content: list[_Block]
    stop_reason: str = "end_turn"
    model: str = "claude-opus-5"
    usage: _Usage = field(default_factory=_Usage)
    stop_details: Any = None


def test_anthropic_maps_thinking_blocks_to_reasoning(anthropic_provider):
    response = _Response(
        content=[_Block("thinking", thinking="considering"), _Block("text", text="done")]
    )
    message = anthropic_provider._to_canonical(response)
    assert [b.type for b in message.content] == ["reasoning", "text"]
    assert message.text() == "done"
    assert message.provider == "anthropic"
    assert message.native[0]["thinking"] == "considering"


def test_anthropic_maps_usage_including_cache_counters(anthropic_provider):
    usage = anthropic_provider._to_usage(_Usage())
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (3, 2)


def test_anthropic_stop_details_are_none_unless_refused(anthropic_provider):
    assert anthropic_provider._stop_details(_Response(content=[])) is None


# --- OpenAI --------------------------------------------------------------


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key")


def test_openai_uses_max_completion_tokens_for_reasoning_models(openai_provider):
    kwargs, _ = openai_provider._build_kwargs(call("gpt-5", [user("hi")], max_tokens=2048))
    assert kwargs["max_completion_tokens"] == 2048
    assert "max_tokens" not in kwargs


def test_openai_uses_max_tokens_for_the_older_models(openai_provider):
    kwargs, _ = openai_provider._build_kwargs(call("gpt-4o", [user("hi")], max_tokens=2048))
    assert kwargs["max_tokens"] == 2048
    assert "max_completion_tokens" not in kwargs


def test_openai_clamps_effort_to_high(openai_provider):
    kwargs, adjustments = openai_provider._build_kwargs(
        call("gpt-5", [user("hi")], effort=Effort.MAX)
    )
    assert kwargs["reasoning_effort"] == "high"
    assert any("clamped" in a for a in adjustments)


def test_openai_drops_top_k_and_says_so(openai_provider):
    kwargs, adjustments = openai_provider._build_kwargs(
        call("gpt-4o", [user("hi")], inference_config=InferenceConfig(top_k=40, top_p=0.9))
    )
    assert "top_k" not in kwargs
    assert kwargs["top_p"] == 0.9
    assert any("top_k" in a for a in adjustments)


def test_openai_reports_that_reasoning_summaries_are_unavailable(openai_provider):
    _, adjustments = openai_provider._build_kwargs(call("gpt-5", [user("hi")], want_reasoning=True))
    assert any("does not return reasoning summaries" in a for a in adjustments)


def test_openai_flattens_the_system_prompt(openai_provider):
    kwargs, _ = openai_provider._build_kwargs(
        call(
            "gpt-5",
            [user("hi")],
            system=[SystemBlock(text="one"), SystemBlock(text="two")],
        )
    )
    assert kwargs["messages"][0] == {"role": "system", "content": "one\n\ntwo"}


def test_openai_turns_a_base64_image_into_a_data_uri(openai_provider):
    message = Message(
        role=Role.USER,
        content=[ImageBlock(source=Base64ImageSource(media_type="image/png", data="AAA"))],
    )
    kwargs, _ = openai_provider._build_kwargs(call("gpt-4o", [message]))
    assert kwargs["messages"][0]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAA"},
    }


def test_openai_passes_an_image_url_through(openai_provider):
    message = Message(
        role=Role.USER,
        content=[ImageBlock(source=UrlImageSource(url="https://example.com/a.png"))],
    )
    kwargs, _ = openai_provider._build_kwargs(call("gpt-4o", [message]))
    assert kwargs["messages"][0]["content"][0]["image_url"]["url"] == ("https://example.com/a.png")


def test_openai_replays_an_anthropic_turn_as_plain_text(openai_provider):
    history = [
        user("q"),
        Message(
            role=Role.ASSISTANT,
            content=[ReasoningBlock(text="hmm"), TextBlock(text="answer")],
            provider="anthropic",
            native=[{"type": "thinking", "thinking": "hmm", "signature": "sig"}],
        ),
        user("q2"),
    ]
    kwargs, _ = openai_provider._build_kwargs(call("gpt-5", history))
    assert kwargs["messages"][1] == {"role": "assistant", "content": "answer"}


@dataclass
class _Msg:
    content: str | None = None
    refusal: str | None = None
    tool_calls: Any = None


@dataclass
class _Choice:
    message: _Msg
    finish_reason: str = "stop"


def test_openai_maps_finish_reasons(openai_provider):
    assert openai_provider._stop_reason(_Choice(_Msg("hi"), "stop")) is StopReason.END_TURN
    assert openai_provider._stop_reason(_Choice(_Msg("hi"), "length")) is StopReason.MAX_TOKENS
    assert openai_provider._stop_reason(_Choice(_Msg("hi"), "tool_calls")) is StopReason.TOOL_USE
    assert openai_provider._stop_reason(_Choice(_Msg(None, "nope"), "stop")) is StopReason.REFUSAL


def test_openai_surfaces_a_refusal_as_the_turn_text(openai_provider):
    message = openai_provider._to_canonical(_Choice(_Msg(None, "I can't help")))
    assert message.text() == "I can't help"
    assert message.native == {"role": "assistant", "content": "I can't help"}


# --- tool translation ----------------------------------------------------


async def _noop(_args):
    return "ok"


CLOCK = ToolDef(
    name="get_time",
    description="read the clock",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_noop,
)


def tool_exchange(provider: str) -> list[Message]:
    """A transcript containing a complete tool call and its result."""
    return [
        user("what time is it?"),
        Message(
            role=Role.ASSISTANT,
            content=[ToolUseBlock(id="tu_1", name="get_time", input={"timezone": "UTC"})],
            provider=provider,
            native=None,
        ),
        Message(
            role=Role.USER,
            content=[ToolResultBlock(tool_use_id="tu_1", name="get_time", content='{"hour": 10}')],
        ),
    ]


def test_anthropic_renders_tool_definitions(anthropic_provider):
    kwargs, _ = anthropic_provider._build_kwargs(call("claude-opus-5", [user("hi")], tools=[CLOCK]))
    assert kwargs["tools"] == [
        {
            "name": "get_time",
            "description": "read the clock",
            "input_schema": CLOCK.input_schema,
        }
    ]


def test_anthropic_keeps_tool_results_on_a_user_message(anthropic_provider):
    kwargs, _ = anthropic_provider._build_kwargs(
        call("claude-opus-5", tool_exchange("anthropic"), tools=[CLOCK])
    )
    messages = kwargs["messages"]
    assert len(messages) == 3

    assert messages[1]["content"][0] == {
        "type": "tool_use",
        "id": "tu_1",
        "name": "get_time",
        "input": {"timezone": "UTC"},
    }
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": '{"hour": 10}',
        "is_error": False,
    }


def test_anthropic_parses_a_tool_use_response(anthropic_provider):
    block = _Block("tool_use")
    block.id, block.name, block.input = "tu_9", "get_time", {"timezone": "UTC"}
    message = anthropic_provider._to_canonical(_Response(content=[block]))

    uses = [b for b in message.content if isinstance(b, ToolUseBlock)]
    assert len(uses) == 1
    assert (uses[0].id, uses[0].name) == ("tu_9", "get_time")
    assert uses[0].input == {"timezone": "UTC"}


def test_openai_renders_tool_definitions_in_function_form(openai_provider):
    kwargs, _ = openai_provider._build_kwargs(call("gpt-5", [user("hi")], tools=[CLOCK]))
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "read the clock",
                "parameters": CLOCK.input_schema,
            },
        }
    ]


def test_openai_fans_tool_results_out_into_tool_role_messages(openai_provider):
    """The shapes genuinely differ: Anthropic keeps results as blocks on a
    user message, OpenAI wants one `tool` message each."""
    kwargs, _ = openai_provider._build_kwargs(call("gpt-5", tool_exchange("openai"), tools=[CLOCK]))
    messages = kwargs["messages"]

    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "tu_1"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "get_time"
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"timezone": "UTC"}

    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "tu_1",
        "content": '{"hour": 10}',
    }
    assert not any(m["role"] == "user" and "tool" in str(m) for m in messages[2:])


def test_an_anthropic_tool_exchange_replays_to_openai(openai_provider):
    """Cross-provider continuity for tool turns, not just text."""
    kwargs, _ = openai_provider._build_kwargs(
        call("gpt-5", tool_exchange("anthropic"), tools=[CLOCK])
    )
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assert kwargs["messages"][1]["tool_calls"][0]["id"] == "tu_1"


def test_openai_parses_tool_calls_from_a_response(openai_provider):
    class _Fn:
        name = "get_time"
        arguments = '{"timezone": "UTC"}'

    class _TC:
        id = "call_1"
        function = _Fn()

    msg = _Msg(None)
    msg.tool_calls = [_TC()]
    message = openai_provider._to_canonical(_Choice(msg, "tool_calls"))

    uses = [b for b in message.content if isinstance(b, ToolUseBlock)]
    assert (uses[0].id, uses[0].name, uses[0].input) == (
        "call_1",
        "get_time",
        {"timezone": "UTC"},
    )
    # The native payload must round-trip for same-provider replay.
    assert message.native["tool_calls"][0]["id"] == "call_1"


def test_openai_survives_malformed_tool_arguments(openai_provider):
    """A model can emit invalid JSON; that must not crash the turn."""

    class _Fn:
        name = "get_time"
        arguments = "{not json"

    class _TC:
        id = "call_2"
        function = _Fn()

    msg = _Msg(None)
    msg.tool_calls = [_TC()]
    message = openai_provider._to_canonical(_Choice(msg, "tool_calls"))

    uses = [b for b in message.content if isinstance(b, ToolUseBlock)]
    assert uses[0].input == {}


def test_neither_adapter_sends_a_tools_key_when_none_are_granted(
    anthropic_provider, openai_provider
):
    a_kwargs, _ = anthropic_provider._build_kwargs(call("claude-opus-5", [user("hi")]))
    o_kwargs, _ = openai_provider._build_kwargs(call("gpt-5", [user("hi")]))
    assert "tools" not in a_kwargs
    assert "tools" not in o_kwargs
