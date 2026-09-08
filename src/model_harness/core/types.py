"""The canonical, provider-neutral wire and domain model.

Everything the service stores and everything it exposes over HTTP is expressed
in these types. Provider adapters translate *into* a vendor SDK's shapes on the
way out and *back into* these types on the way in; nothing vendor-specific
leaks into the session store except the opaque ``native`` field on
:class:`Message`, which exists purely so a same-provider replay is lossless.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(StrEnum):
    """Normalized termination reason.

    Anthropic's set is the superset, so it is the basis; OpenAI's
    ``finish_reason`` values are mapped onto it by the OpenAI adapter.
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    REFUSAL = "refusal"


class Effort(StrEnum):
    """Neutral reasoning-depth knob.

    Anthropic accepts all five verbatim via ``output_config.effort``. OpenAI's
    ``reasoning_effort`` has no level above ``high``, so the OpenAI adapter
    clamps ``xhigh`` and ``max`` down to ``high`` and reports the clamp in the
    response's ``adjustments`` list.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


# --- Content blocks -------------------------------------------------------


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Base64ImageSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class UrlImageSource(BaseModel):
    type: Literal["url"] = "url"
    url: str


ImageSource = Annotated[
    Base64ImageSource | UrlImageSource,
    Field(discriminator="type"),
]


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSource


class ReasoningBlock(BaseModel):
    """A model's visible reasoning summary.

    Only ever produced by the service, never accepted from a client. The raw
    chain of thought is not exposed by either provider; what lands here is the
    summary the provider chose to return, which may be an empty string.
    """

    type: Literal["reasoning"] = "reasoning"
    text: str = ""


class ToolUseBlock(BaseModel):
    """A tool call the model asked for.

    ``id`` is the provider's correlation id. It is echoed back on the matching
    result and must survive round-tripping, because both providers reject a
    result whose id does not match an outstanding call.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """The outcome of executing one tool call.

    Carried on a ``user`` message, matching Anthropic's shape. The OpenAI
    adapter splits these into its own ``tool``-role messages on the way out —
    keeping one canonical form is what lets a transcript containing tool calls
    replay against either provider.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    name: str = ""
    content: str = ""
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ImageBlock | ReasoningBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """One conversation turn in canonical form.

    ``provider`` and ``native`` are populated for assistant turns the service
    generated. They are excluded from HTTP responses (``exclude`` on the field)
    and exist only so an adapter can replay its own earlier turn verbatim
    instead of a lossy canonical rendering — see each adapter's
    ``_to_native_message``.
    """

    model_config = ConfigDict(populate_by_name=True)

    role: Role
    content: list[ContentBlock]

    provider: str | None = Field(default=None, exclude=True)
    native: Any | None = Field(default=None, exclude=True)

    def text(self) -> str:
        """Concatenate the message's text blocks, ignoring reasoning and images."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))


class SystemBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


# --- Request / response ---------------------------------------------------


class InferenceConfig(BaseModel):
    """Neutral generation parameters.

    A parameter a target model does not accept is dropped rather than
    forwarded, and the drop is reported in the response's ``adjustments``. This
    This matters most for sampling: no Anthropic model gets these (the
    frontier models reject them, and the SDK dropped the parameters in 1.x),
    and OpenAI's reasoning models reject them too. Forwarding a client's
    harmless default would turn a working request into a 400.
    """

    max_tokens: int | None = Field(default=None, gt=0, le=128_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    stop_sequences: list[str] | None = Field(default=None, max_length=8)


class ConverseRequest(BaseModel):
    model: str = Field(
        min_length=1, description="Harness model id, e.g. 'claude-opus-5' or 'gpt-5'"
    )

    session_id: str | None = Field(
        default=None,
        description=(
            "Existing session to continue. Omit to start a new one; the id of the "
            "new session is returned in the response and in the X-Session-Id header."
        ),
    )

    messages: list[Message] = Field(
        min_length=1,
        description=(
            "The new turn(s) only, not the whole history. The service prepends the "
            "stored transcript for the session before calling the provider."
        ),
    )

    system: list[SystemBlock] | None = Field(
        default=None,
        description=(
            "System prompt. Recorded on the session when it is created. Sending a "
            "different system prompt on a later turn of the same session replaces "
            "it, which invalidates the provider's prompt cache for that session."
        ),
    )

    tools: list[str] | None = Field(
        default=None,
        max_length=32,
        description=(
            "Names of tools to expose for this turn, from GET /v1/tools. Omit "
            "for a plain completion with no tools. Each name must also be "
            "permitted for the calling principal."
        ),
    )
    max_tool_iterations: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Cap on model->tool->model round trips for this turn. The loop also "
            "stops as soon as the model answers without calling a tool."
        ),
    )

    inference_config: InferenceConfig = Field(default_factory=InferenceConfig)
    effort: Effort | None = None
    stream_reasoning: bool = Field(
        default=False,
        description="Ask the provider for a reasoning summary instead of omitting it.",
    )
    metadata: dict[str, str] = Field(default_factory=dict, max_length=16)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolInvocation(BaseModel):
    """One executed tool call, reported back for observability.

    A caller cannot see the loop happening, so without this a turn that ran
    six tool calls is indistinguishable from one that ran none.
    """

    name: str
    tool_use_id: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    is_error: bool = False
    duration_ms: int = 0


class ConverseOutput(BaseModel):
    message: Message


class ConverseResponse(BaseModel):
    session_id: str
    provider: str
    model: str
    """The harness model id that was requested."""
    native_model: str
    """The id the provider reported serving, which can differ (fallbacks, aliases)."""

    output: ConverseOutput
    stop_reason: StopReason
    stop_details: dict[str, Any] | None = None
    usage: Usage
    latency_ms: int
    tool_calls: list[ToolInvocation] = Field(
        default_factory=list,
        description="Tools executed while producing this turn, in order.",
    )
    iterations: int = Field(
        default=1, description="Provider round trips, including the final answer."
    )
    adjustments: list[str] = Field(
        default_factory=list,
        description="Neutral parameters the target model could not accept verbatim.",
    )


# --- Streaming ------------------------------------------------------------


class StreamEvent(BaseModel):
    """One canonical SSE payload.

    Deliberately flatter than either provider's event model: a client that can
    render these can render any provider the harness gains later.
    """

    type: Literal[
        "message_start",
        "content_block_start",
        "content_delta",
        "content_block_stop",
        "message_stop",
        "error",
    ]

    session_id: str | None = None
    provider: str | None = None
    model: str | None = None
    native_model: str | None = None

    index: int | None = None
    block_type: Literal["text", "reasoning"] | None = None
    text: str | None = None

    stop_reason: StopReason | None = None
    usage: Usage | None = None
    adjustments: list[str] | None = None

    code: str | None = None
    message: str | None = None
    error_id: str | None = None
