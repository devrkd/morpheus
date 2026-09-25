"""The model catalog — the policy layer we kept.

Strands will talk to any model string handed to it. The catalog is what makes
a request *allowed*: which ids we accept, which aliases map where, and what
each model can actually take. Routing has to be answerable without a network
call, so it stays static.
"""

from __future__ import annotations

from morpheus.core.registry import Provider, ThinkingStyle, known_ids, resolve


def test_resolves_canonical_ids_and_aliases():
    assert resolve("claude-opus-5").provider is Provider.ANTHROPIC
    assert resolve("gpt-5").provider is Provider.OPENAI
    assert resolve("opus").id == "claude-opus-5"
    assert resolve("GPT5").id == "gpt-5"
    assert resolve("  claude-sonnet-5  ").id == "claude-sonnet-5"


def test_unknown_model_resolves_to_none():
    assert resolve("llama-9") is None
    assert "claude-opus-5" in known_ids()


def test_native_ids_are_what_the_provider_is_sent():
    """Strands passes model_id through, so ours must be the provider's."""
    for model_id in known_ids():
        spec = resolve(model_id)
        assert spec.native_id
        # Anthropic ids are complete as written — never date-suffixed.
        if spec.provider is Provider.ANTHROPIC:
            assert not spec.native_id[-1].isdigit() or "-" in spec.native_id


def test_capability_flags_describe_what_a_model_accepts():
    """These drive the `adjustments` a caller gets when a parameter cannot be
    forwarded, so they have to stay accurate."""
    opus = resolve("claude-opus-5")
    assert opus.supports_sampling is False  # rejected by the frontier models
    assert opus.supports_effort is True

    haiku = resolve("claude-haiku-4-5")
    assert haiku.supports_effort is False  # errors on Haiku 4.5

    gpt5 = resolve("gpt-5")
    assert gpt5.supports_sampling is False  # reasoning models reject it
    assert gpt5.max_effort.value == "high"  # no level above high

    gpt4o = resolve("gpt-4o")
    assert gpt4o.supports_sampling is True


def test_reasoning_style_matches_the_provider_family():
    assert resolve("claude-opus-5").thinking is ThinkingStyle.ADAPTIVE
    assert resolve("gpt-5").thinking is ThinkingStyle.REASONING_EFFORT
    assert resolve("gpt-4o").thinking is ThinkingStyle.NONE
