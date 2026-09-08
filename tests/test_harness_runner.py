"""The Strands-backed runner.

Driven by a scripted Strands ``Model``, so the real agent loop, real tool
execution and real session files are exercised with no credential and no
spend. The point of these tests is that the migration keeps the policy
properties Strands knows nothing about: model allowlists, tool grants, and
session ownership.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from strands.models.model import Model

from model_harness.auth.principals import Principal, hash_key
from model_harness.config import Settings
from model_harness.core.types import (
    ConverseRequest,
    Message,
    Role,
    StopReason,
    SystemBlock,
    TextBlock,
)
from model_harness.errors import (
    InvalidRequestError,
    ModelNotPermittedError,
    SessionNotFoundError,
    ToolNotPermittedError,
    UnknownModelError,
)
from model_harness.harness.models import ModelFactory
from model_harness.harness.ownership import OwnershipIndex
from model_harness.harness.runner import AgentRunner


class ScriptedModel(Model):
    """Replays turns as Strands stream events. Each turn is text or a tool call."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.seen: list[list[dict]] = []

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
def settings() -> Settings:
    return Settings(
        ANTHROPIC_API_KEY="test-anthropic",
        OPENAI_API_KEY="test-openai",
        HARNESS_ALLOW_ANONYMOUS=True,
        _env_file=None,
    )


@pytest.fixture
def alice() -> Principal:
    return Principal(id="alice", key_sha256=hash_key("a"), allowed_tools=frozenset({"*"}))


@pytest.fixture
def bob() -> Principal:
    return Principal(id="bob", key_sha256=hash_key("b"))


def build(tmp_path: Path, settings: Settings, model: Model) -> AgentRunner:
    return AgentRunner(
        settings=settings,
        factory=ModelFactory(settings),
        ownership=OwnershipIndex(tmp_path / "index.json"),
        session_dir=tmp_path / "sessions",
        model_override=model,
    )


def req(text="hello", model="claude-opus-5", session_id=None, tools=None, system=None):
    return ConverseRequest(
        model=model,
        session_id=session_id,
        messages=[Message(role=Role.USER, content=[TextBlock(text=text)])],
        tools=tools,
        system=[SystemBlock(text=system)] if system else None,
    )


# --- the loop, via Strands ------------------------------------------------


async def test_a_plain_turn_returns_our_canonical_shape(tmp_path, settings, alice):
    runner = build(tmp_path, settings, ScriptedModel([{"text": "hi there"}]))
    result = await runner.converse(req(), alice)

    assert result.output.message.text() == "hi there"
    assert result.provider == "anthropic"
    assert result.model == "claude-opus-5"
    assert result.stop_reason is StopReason.END_TURN
    assert result.usage.input_tokens == 11
    assert result.session_id.startswith("sess_")


async def test_a_tool_turn_runs_the_tool_and_loops(tmp_path, settings, alice):
    model = ScriptedModel(
        [
            {"tool": ("get_current_time", {"timezone": "UTC"}, "tu_1")},
            {"text": "it is 09:00"},
        ]
    )
    runner = build(tmp_path, settings, model)
    result = await runner.converse(req(tools=["get_current_time"]), alice)

    assert result.output.message.text() == "it is 09:00"
    assert result.iterations == 2
    assert [c.name for c in result.tool_calls] == ["get_current_time"]
    # Usage is summed across the loop, not just the final call.
    assert result.usage.input_tokens == 22


async def test_tool_schemas_are_generated_from_type_hints(tmp_path, settings, alice):
    """Strands derives the JSON Schema, so we no longer hand-write them."""
    model = ScriptedModel([{"text": "ok"}])
    runner = build(tmp_path, settings, model)
    await runner.converse(req(tools=["get_current_time", "http_request"]), alice)

    names = {spec["name"] for spec in model.last_tool_specs}
    assert names == {"get_current_time", "http_request"}
    for spec in model.last_tool_specs:
        assert spec["description"]
        assert spec["inputSchema"]


async def test_no_tools_requested_means_none_are_offered(tmp_path, settings, alice):
    model = ScriptedModel([{"text": "ok"}])
    runner = build(tmp_path, settings, model)
    await runner.converse(req(), alice)
    assert not model.last_tool_specs


async def test_the_system_prompt_reaches_the_model(tmp_path, settings, alice):
    model = ScriptedModel([{"text": "ok"}])
    runner = build(tmp_path, settings, model)
    await runner.converse(req(system="Be terse."), alice)
    assert model.last_system_prompt == "Be terse."


# --- session continuity, now Strands' job ---------------------------------


async def test_history_is_restored_on_the_next_turn(tmp_path, settings, alice):
    runner = build(tmp_path, settings, ScriptedModel([{"text": "noted"}]))
    first = await runner.converse(req("my name is Ramesh"), alice)

    second_model = ScriptedModel([{"text": "Ramesh"}])
    runner2 = AgentRunner(
        settings=settings,
        factory=ModelFactory(settings),
        ownership=OwnershipIndex(tmp_path / "index.json"),
        session_dir=tmp_path / "sessions",
        model_override=second_model,
    )
    await runner2.converse(req("what is my name?", session_id=first.session_id), alice)

    # The second model saw turn one replayed from disk, by a different runner
    # instance — the property the whole service exists for.
    replayed = second_model.seen[0]
    texts = [
        c.get("text", "") for m in replayed for c in m.get("content", []) if isinstance(c, dict)
    ]
    assert any("my name is Ramesh" in t for t in texts)
    assert any("noted" in t for t in texts)


async def test_transcripts_land_on_disk(tmp_path, settings, alice):
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    result = await runner.converse(req(), alice)

    files = list((tmp_path / "sessions").rglob("*.json"))
    assert files, "expected Strands to persist the session"
    assert any(result.session_id in str(f) for f in files)


# --- the properties Strands knows nothing about ---------------------------


async def test_another_principal_cannot_open_your_session(tmp_path, settings, alice, bob):
    """Strands would happily load any session id handed to it. Authorization
    stays on our side, or this hole reopens on migration."""
    runner = build(tmp_path, settings, ScriptedModel([{"text": "secret is 42"}]))
    first = await runner.converse(req("remember 42"), alice)

    with pytest.raises(SessionNotFoundError):
        await runner.converse(req("what is it?", session_id=first.session_id), bob)


async def test_session_listing_is_scoped_to_the_owner(tmp_path, settings, alice, bob):
    index = OwnershipIndex(tmp_path / "index.json")
    runner = AgentRunner(
        settings=settings,
        factory=ModelFactory(settings),
        ownership=index,
        session_dir=tmp_path / "sessions",
        model_override=ScriptedModel([{"text": "ok"}, {"text": "ok"}]),
    )
    mine = await runner.converse(req("a"), alice)
    theirs = await runner.converse(req("b"), bob)

    assert [r.session_id for r in await index.list_for("alice")] == [mine.session_id]
    assert [r.session_id for r in await index.list_for("bob")] == [theirs.session_id]


async def test_a_forbidden_model_is_refused(tmp_path, settings):
    restricted = Principal(
        id="sonnet-only",
        key_sha256=hash_key("s"),
        allowed_models=frozenset({"claude-sonnet-5"}),
    )
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    with pytest.raises(ModelNotPermittedError):
        await runner.converse(req(model="claude-opus-5"), restricted)


async def test_an_alias_cannot_bypass_the_model_allowlist(tmp_path, settings):
    restricted = Principal(
        id="sonnet-only",
        key_sha256=hash_key("s"),
        allowed_models=frozenset({"claude-sonnet-5"}),
    )
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    with pytest.raises(ModelNotPermittedError):
        await runner.converse(req(model="opus"), restricted)


async def test_a_dangerous_tool_is_denied_by_default(tmp_path, settings, bob):
    """`bob` has no allowed_tools, so he gets safe tools only."""
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    with pytest.raises(ToolNotPermittedError, match="granted explicitly"):
        await runner.converse(req(tools=["http_request"]), bob)


async def test_a_safe_tool_is_granted_by_default(tmp_path, settings, bob):
    model = ScriptedModel([{"text": "ok"}])
    runner = build(tmp_path, settings, model)
    await runner.converse(req(tools=["get_current_time"]), bob)
    assert [s["name"] for s in model.last_tool_specs] == ["get_current_time"]


async def test_unknown_model_and_unknown_tool_are_reported(tmp_path, settings, alice):
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    with pytest.raises(UnknownModelError):
        await runner.converse(req(model="gemini-ultra"), alice)
    with pytest.raises(InvalidRequestError, match="Unknown tool"):
        await runner.converse(req(tools=["nope"]), alice)


async def test_a_failed_first_turn_leaves_no_empty_session(tmp_path, settings, alice):
    class Exploding(ScriptedModel):
        async def stream(self, *a, **kw):
            raise RuntimeError("provider down")
            yield {}

    index = OwnershipIndex(tmp_path / "index.json")
    runner = AgentRunner(
        settings=settings,
        factory=ModelFactory(settings),
        ownership=index,
        session_dir=tmp_path / "sessions",
        model_override=Exploding(),
    )
    with pytest.raises(RuntimeError):
        await runner.converse(req(), alice)

    assert await index.list_for("alice") == []


async def test_the_ownership_index_survives_a_restart(tmp_path, settings, alice):
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    first = await runner.converse(req("hello there"), alice)

    # A fresh index, as a restarted process would build.
    reloaded = OwnershipIndex(tmp_path / "index.json")
    record = await reloaded.get(first.session_id, "alice")
    assert record is not None
    assert record.owner == "alice"
    assert record.preview == "hello there"
    assert record.turn_count == 1
    # ...and still not readable by anyone else.
    assert await reloaded.get(first.session_id, "bob") is None
