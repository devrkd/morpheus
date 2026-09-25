"""Shared fixtures.

Everything is driven by a scripted Strands ``Model``, so the real agent loop,
real tool execution and real session files are exercised with no credential
and no network.
"""

from __future__ import annotations

import json

import pytest
from strands.models.model import Model

from morpheus.auth.principals import Principal, PrincipalStore, hash_key
from morpheus.config import Settings

ALICE_KEY = "mh_test_alice_key"
BOB_KEY = "mh_test_bob_key"
SONNET_ONLY_KEY = "mh_test_sonnet_only_key"
DISABLED_KEY = "mh_test_disabled_key"


class ScriptedModel(Model):
    """Replays turns as Strands stream events. Each turn is text or a tool call."""

    def __init__(self, script=None):
        # Held by reference, not copied, so a test can rewrite the script after
        # the app (and therefore this model) has been constructed.
        self.script = script if script is not None else []
        self.seen: list[list[dict]] = []
        self.last_tool_specs: list[dict] | None = None
        self.last_system_prompt: str | None = None

    def get_config(self):
        return {}

    def update_config(self, **kw):
        pass

    async def structured_output(self, output_model, prompt, system_prompt=None, **kw):
        yield {}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        self.seen.append([dict(m) for m in messages])
        self.last_tool_specs = tool_specs
        self.last_system_prompt = system_prompt
        turn = self.script.pop(0) if self.script else {"text": "done"}

        yield {"messageStart": {"role": "assistant"}}
        if "tool" in turn:
            name, args, tuid = turn["tool"]
            yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": tuid}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": turn["text"]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
                "metrics": {"latencyMs": 5},
            }
        }


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        ANTHROPIC_API_KEY="test-anthropic",
        OPENAI_API_KEY="test-openai",
        HARNESS_ALLOW_ANONYMOUS=True,
        HARNESS_SESSION_DIR=str(tmp_path / "sessions"),
        _env_file=None,
    )


@pytest.fixture
def alice() -> Principal:
    return Principal(id="alice", key_sha256=hash_key(ALICE_KEY), allowed_tools=frozenset({"*"}))


@pytest.fixture
def bob() -> Principal:
    return Principal(id="bob", key_sha256=hash_key(BOB_KEY))


@pytest.fixture
def sonnet_only() -> Principal:
    return Principal(
        id="sonnet-only",
        key_sha256=hash_key(SONNET_ONLY_KEY),
        allowed_models=frozenset({"claude-sonnet-5"}),
        max_tokens_per_turn=500,
    )


@pytest.fixture
def principal_store(alice, bob, sonnet_only) -> PrincipalStore:
    return PrincipalStore(
        [
            alice,
            bob,
            sonnet_only,
            Principal(id="retired", key_sha256=hash_key(DISABLED_KEY), disabled=True),
        ]
    )
