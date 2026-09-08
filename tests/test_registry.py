from __future__ import annotations

from model_harness.core.registry import Provider, ThinkingStyle, known_ids, resolve
from model_harness.core.types import Effort, InferenceConfig
from model_harness.providers.base import resolve_effort, sampling_params


def test_resolves_canonical_ids_and_aliases():
    assert resolve("claude-opus-5").provider is Provider.ANTHROPIC
    assert resolve("gpt-5").provider is Provider.OPENAI
    assert resolve("opus").id == "claude-opus-5"
    assert resolve("GPT5").id == "gpt-5"
    assert resolve("  claude-sonnet-5  ").id == "claude-sonnet-5"


def test_unknown_model_resolves_to_none():
    assert resolve("llama-9") is None
    assert "claude-opus-5" in known_ids()


def test_sampling_is_dropped_for_models_that_reject_it():
    cfg = InferenceConfig(temperature=0.7, top_p=0.9)

    params, notes = sampling_params(cfg, resolve("claude-opus-5"))
    assert params == {}
    assert notes and "temperature" in notes[0] and "top_p" in notes[0]

    params, notes = sampling_params(cfg, resolve("gpt-4o"))
    assert params == {"temperature": 0.7, "top_p": 0.9}
    assert notes == []


def test_sampling_notes_are_silent_when_nothing_was_requested():
    params, notes = sampling_params(InferenceConfig(), resolve("claude-opus-5"))
    assert (params, notes) == ({}, [])


def test_effort_is_clamped_or_dropped_per_model():
    assert resolve_effort(Effort.MAX, resolve("claude-opus-5")) == (Effort.MAX, [])

    effort, notes = resolve_effort(Effort.MAX, resolve("gpt-5"))
    assert effort is Effort.HIGH
    assert "clamped" in notes[0]

    effort, notes = resolve_effort(Effort.HIGH, resolve("claude-haiku-4-5"))
    assert effort is None
    assert "dropped effort" in notes[0]

    assert resolve_effort(None, resolve("gpt-5")) == (None, [])


def test_reasoning_style_matches_provider_family():
    assert resolve("claude-opus-5").thinking is ThinkingStyle.ADAPTIVE
    assert resolve("gpt-5").thinking is ThinkingStyle.REASONING_EFFORT
    assert resolve("gpt-4o").thinking is ThinkingStyle.NONE
