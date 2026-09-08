"""Model catalog: the routing table and the capability table in one place.

A harness model id maps to exactly one provider. The capability flags on each
entry are what let the service *drop* a neutral parameter the target model
would reject, instead of forwarding it and turning a benign default into a 400.

Keeping this static rather than discovered at startup is deliberate: routing
must be answerable without a network call, and a model the harness has never
been told about should fail fast with a clear error rather than be guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .types import Effort


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ThinkingStyle(StrEnum):
    ADAPTIVE = "adaptive"
    """Anthropic 4.6+: `thinking={"type": "adaptive"}`, depth via `output_config.effort`."""

    REASONING_EFFORT = "reasoning_effort"
    """OpenAI reasoning models: depth via `reasoning_effort`, no budget parameter."""

    NONE = "none"
    """No server-side reasoning control; the harness sends neither knob."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    """The id clients send to the harness."""

    provider: Provider
    native_id: str
    """The id sent to the provider. Differs where a provider namespaces its ids."""

    context_window: int
    max_output_tokens: int
    thinking: ThinkingStyle

    supports_sampling: bool
    """Whether the harness forwards `temperature`/`top_p`/`top_k`.

    False for every Anthropic model and for OpenAI's reasoning models. Two
    different reasons converge on the same answer: the current Anthropic
    frontier models reject these parameters with a 400, and the `anthropic`
    Python SDK dropped them from `messages.create` in its 1.x major version, so
    there is no first-class way to send them even to an older model that would
    still accept them. Requests carrying sampling parameters for these models
    have them dropped, with the drop reported in the response's `adjustments`.
    """

    supports_stop_sequences: bool = True
    supports_images: bool = True

    supports_effort: bool = True
    """Whether the model accepts a reasoning-depth knob at all."""

    max_effort: Effort = Effort.MAX
    """Highest effort level the provider accepts; higher requests are clamped."""

    aliases: tuple[str, ...] = field(default=())


_EFFORT_ORDER: dict[Effort, int] = {
    Effort.LOW: 0,
    Effort.MEDIUM: 1,
    Effort.HIGH: 2,
    Effort.XHIGH: 3,
    Effort.MAX: 4,
}


def effort_rank(effort: Effort) -> int:
    return _EFFORT_ORDER[effort]


# Anthropic ids are complete as written — never append a date suffix.
_ANTHROPIC_MODELS = [
    ModelSpec(
        id="claude-opus-5",
        provider=Provider.ANTHROPIC,
        native_id="claude-opus-5",
        context_window=1_000_000,
        max_output_tokens=128_000,
        thinking=ThinkingStyle.ADAPTIVE,
        supports_sampling=False,
        aliases=("opus", "claude-opus"),
    ),
    ModelSpec(
        id="claude-opus-4-8",
        provider=Provider.ANTHROPIC,
        native_id="claude-opus-4-8",
        context_window=1_000_000,
        max_output_tokens=128_000,
        thinking=ThinkingStyle.ADAPTIVE,
        supports_sampling=False,
    ),
    ModelSpec(
        id="claude-sonnet-5",
        provider=Provider.ANTHROPIC,
        native_id="claude-sonnet-5",
        context_window=1_000_000,
        max_output_tokens=128_000,
        thinking=ThinkingStyle.ADAPTIVE,
        supports_sampling=False,
        aliases=("sonnet", "claude-sonnet"),
    ),
    ModelSpec(
        id="claude-haiku-4-5",
        provider=Provider.ANTHROPIC,
        native_id="claude-haiku-4-5",
        context_window=200_000,
        # Conservative harness-side clamp, not a published provider ceiling.
        max_output_tokens=32_000,
        # Haiku 4.5 predates adaptive thinking and takes a `budget_tokens`
        # budget instead. Rather than invent a budget per request, the harness
        # runs it without server-side thinking; use a 4.6+ model if you need it.
        thinking=ThinkingStyle.NONE,
        # The API would accept sampling here, but the SDK has no parameter for
        # it since 1.x — see ModelSpec.supports_sampling.
        supports_sampling=False,
        supports_effort=False,  # `effort` is rejected on Haiku 4.5.
        aliases=("haiku",),
    ),
]

_OPENAI_MODELS = [
    ModelSpec(
        id="gpt-5",
        provider=Provider.OPENAI,
        native_id="gpt-5",
        context_window=400_000,
        max_output_tokens=128_000,
        thinking=ThinkingStyle.REASONING_EFFORT,
        supports_sampling=False,
        # OpenAI's reasoning_effort tops out at "high"; xhigh/max clamp down.
        max_effort=Effort.HIGH,
        aliases=("gpt5",),
    ),
    ModelSpec(
        id="gpt-5-mini",
        provider=Provider.OPENAI,
        native_id="gpt-5-mini",
        context_window=400_000,
        max_output_tokens=128_000,
        thinking=ThinkingStyle.REASONING_EFFORT,
        supports_sampling=False,
        max_effort=Effort.HIGH,
    ),
    ModelSpec(
        id="gpt-4.1",
        provider=Provider.OPENAI,
        native_id="gpt-4.1",
        context_window=1_000_000,
        max_output_tokens=32_768,
        thinking=ThinkingStyle.NONE,
        supports_sampling=True,
        supports_effort=False,
    ),
    ModelSpec(
        id="gpt-4o",
        provider=Provider.OPENAI,
        native_id="gpt-4o",
        context_window=128_000,
        max_output_tokens=16_384,
        thinking=ThinkingStyle.NONE,
        supports_sampling=True,
        supports_effort=False,
    ),
]

MODELS: dict[str, ModelSpec] = {}
_ALIASES: dict[str, str] = {}

for _spec in (*_ANTHROPIC_MODELS, *_OPENAI_MODELS):
    MODELS[_spec.id] = _spec
    for _alias in _spec.aliases:
        _ALIASES[_alias] = _spec.id


def resolve(model_id: str) -> ModelSpec | None:
    """Look up a model by id or alias. Returns None for an unknown id."""
    key = model_id.strip()
    if key in MODELS:
        return MODELS[key]
    aliased = _ALIASES.get(key.lower())
    return MODELS[aliased] if aliased else None


def known_ids() -> list[str]:
    return sorted(MODELS)
