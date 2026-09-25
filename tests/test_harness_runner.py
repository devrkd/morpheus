"""The Strands-backed runner.

Driven by a scripted Strands ``Model``, so the real agent loop, real tool
execution and real session files are exercised with no credential and no
spend. The point of these tests is that the migration keeps the policy
properties Strands knows nothing about: model allowlists, tool grants, and
session ownership.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from strands.models.model import Model

from morpheus.auth.principals import Principal, hash_key
from morpheus.config import Settings
from morpheus.core.types import (
    ConverseRequest,
    Message,
    Role,
    StopReason,
    SystemBlock,
    TextBlock,
)
from morpheus.errors import (
    InvalidRequestError,
    ModelNotPermittedError,
    SessionNotFoundError,
    ToolNotPermittedError,
    UnknownModelError,
)
from morpheus.harness.models import ModelFactory
from morpheus.harness.ownership import OwnershipIndex
from morpheus.harness.runner import AgentRunner

from .conftest import ScriptedModel


def build(tmp_path: Path, settings: Settings, model: Model) -> AgentRunner:
    return AgentRunner(
        settings=settings,
        factory=ModelFactory(settings),
        ownership=OwnershipIndex(tmp_path / "index.json"),
        session_dir=tmp_path / "sessions",
        model_override=model,
    )


def req(
    text="hello",
    model="claude-opus-5",
    session_id=None,
    tools=None,
    system=None,
    effort=None,
    max_tool_iterations=None,
):
    return ConverseRequest(
        model=model,
        session_id=session_id,
        messages=[Message(role=Role.USER, content=[TextBlock(text=text)])],
        tools=tools,
        system=[SystemBlock(text=system)] if system else None,
        effort=effort,
        max_tool_iterations=max_tool_iterations,
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


# --- the honesty contract, after the migration ----------------------------


async def test_sampling_is_dropped_for_models_that_reject_it(tmp_path, settings, alice):
    """The frontier Anthropic models and OpenAI's reasoning models answer a
    stray temperature with a 400, so forwarding a caller's harmless default
    would break a request that ought to succeed."""
    from morpheus.core.types import InferenceConfig

    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    result = await runner.converse(
        ConverseRequest(
            model="claude-opus-5",
            messages=[Message(role=Role.USER, content=[TextBlock(text="hi")])],
            inference_config=InferenceConfig(temperature=0.5, top_p=0.9),
        ),
        alice,
    )
    joined = " | ".join(result.adjustments)
    assert "temperature" in joined and "top_p" in joined
    assert "not accepted by claude-opus-5" in joined


def test_sampling_reaches_a_model_that_accepts_it(settings):
    """Verified on the real provider config, not just reported."""
    from morpheus.core.registry import resolve as resolve_model
    from morpheus.harness.models import ModelFactory

    factory = ModelFactory(settings)
    model = factory.build(resolve_model("gpt-4o"), max_tokens=2048, sampling={"temperature": 0.3})
    params = model.get_config()["params"]
    assert params["temperature"] == 0.3
    assert params["max_tokens"] == 2048


async def test_top_k_is_dropped_for_openai_only(tmp_path, settings, alice):
    from morpheus.core.types import InferenceConfig

    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    result = await runner.converse(
        ConverseRequest(
            model="gpt-4o",
            messages=[Message(role=Role.USER, content=[TextBlock(text="hi")])],
            inference_config=InferenceConfig(top_k=40, temperature=0.2),
        ),
        alice,
    )
    assert any("top_k" in a and "no equivalent" in a for a in result.adjustments)


async def test_max_tokens_is_capped_to_the_model_ceiling(tmp_path, settings, alice):
    from morpheus.core.types import InferenceConfig

    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    result = await runner.converse(
        ConverseRequest(
            model="gpt-4o",
            messages=[Message(role=Role.USER, content=[TextBlock(text="hi")])],
            inference_config=InferenceConfig(max_tokens=100_000),
        ),
        alice,
    )
    assert any("lowered to 16384" in a for a in result.adjustments)


async def test_a_principal_ceiling_is_enforced_not_just_declared(tmp_path, settings):
    """It used to be reported as unenforced. It is now applied."""
    capped = Principal(id="capped", key_sha256=hash_key("c"), max_tokens_per_turn=500)
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    result = await runner.converse(req(), capped)
    assert any("per-turn ceiling for principal 'capped'" in a for a in result.adjustments)


def test_the_principal_ceiling_reaches_the_provider_config(settings):
    from morpheus.core.registry import resolve as resolve_model
    from morpheus.harness.models import ModelFactory

    factory = ModelFactory(settings)
    model = factory.build(resolve_model("claude-opus-5"), max_tokens=500)
    assert model.get_config()["max_tokens"] == 500


async def test_effort_is_declared_as_not_forwarded(tmp_path, settings, alice):
    """Anthropic exposes it as output_config.effort and OpenAI as
    reasoning_effort; neither is reachable through the Strands config yet, so
    it is declared rather than silently ignored."""
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    result = await runner.converse(req(effort="high"), alice)
    assert any("dropped effort" in a for a in result.adjustments)


async def test_a_clean_request_reports_nothing(tmp_path, settings, alice):
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    result = await runner.converse(req(), alice)
    assert result.adjustments == []


# --- streaming ------------------------------------------------------------


async def test_streaming_yields_tool_lifecycle_then_text(tmp_path, settings, alice):
    model = ScriptedModel(
        [
            {"tool": ("get_current_time", {"timezone": "UTC"}, "tu_1")},
            {"text": "It is 09:00."},
        ]
    )
    runner = build(tmp_path, settings, model)
    session_id, events = await runner.converse_stream(req(tools=["get_current_time"]), alice)

    collected = [e async for e in events]
    kinds = [e.type for e in collected]

    assert kinds[0] == "message_start"
    assert kinds[-1] == "message_stop"
    assert "tool_start" in kinds and "tool_end" in kinds
    # The tool call must be announced before any text, or the client shows a
    # frozen connection while the tool runs.
    assert kinds.index("tool_start") < kinds.index("content_delta")

    start = next(e for e in collected if e.type == "tool_start")
    end = next(e for e in collected if e.type == "tool_end")
    assert start.tool_name == "get_current_time"
    assert start.tool_use_id == end.tool_use_id == "tu_1"
    assert end.is_error is False
    assert end.tool_output is None  # withheld unless asked for

    stop = collected[-1]
    assert stop.session_id == session_id
    assert stop.iterations == 2


async def test_a_tool_call_is_announced_once_not_per_delta(tmp_path, settings, alice):
    """Strands repeats `current_tool_use` while the arguments accumulate."""
    model = ScriptedModel(
        [
            {"tool": ("get_current_time", {"timezone": "UTC"}, "tu_1")},
            {"text": "done"},
        ]
    )
    runner = build(tmp_path, settings, model)
    _, events = await runner.converse_stream(req(tools=["get_current_time"]), alice)
    starts = [e for e in [x async for x in events] if e.type == "tool_start"]
    assert len(starts) == 1


async def test_streaming_persists_the_turn_and_records_it(tmp_path, settings, alice):
    index = OwnershipIndex(tmp_path / "index.json")
    runner = AgentRunner(
        settings=settings,
        factory=ModelFactory(settings),
        ownership=index,
        session_dir=tmp_path / "sessions",
        model_override=ScriptedModel([{"text": "streamed"}]),
    )
    session_id, events = await runner.converse_stream(req("stream me"), alice)
    [_ async for _ in events]

    record = await index.get(session_id, "alice")
    assert record is not None
    assert record.turn_count == 1
    assert record.preview == "stream me"
    assert await runner.read_transcript(session_id)


async def test_streaming_enforces_session_ownership(tmp_path, settings, alice, bob):
    runner = build(tmp_path, settings, ScriptedModel([{"text": "ok"}]))
    first = await runner.converse(req(), alice)
    with pytest.raises(SessionNotFoundError):
        await runner.converse_stream(req(session_id=first.session_id), bob)


async def test_an_abandoned_stream_leaves_no_empty_session(tmp_path, settings, alice):
    """A client hang-up mid-stream must not litter the session list."""
    index = OwnershipIndex(tmp_path / "index.json")
    runner = AgentRunner(
        settings=settings,
        factory=ModelFactory(settings),
        ownership=index,
        session_dir=tmp_path / "sessions",
        model_override=ScriptedModel([{"text": "never read"}]),
    )
    session_id, events = await runner.converse_stream(req(), alice)

    gen = events.__aiter__()
    await gen.__anext__()  # message_start only, then walk away
    await events.aclose()

    assert await index.get(session_id, "alice") is None


# --- the loop must be bounded --------------------------------------------


class NeverStopsModel(ScriptedModel):
    """Asks for a tool on every turn, forever.

    Reproduces an apparently stuck request: Strands treats an omitted cap as
    *no limit*, so before `limits` was passed this ran until something else
    gave out.
    """

    def __init__(self):
        super().__init__([])
        self.turns = 0

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        self.turns += 1
        self.last_tool_specs = tool_specs
        self.last_system_prompt = system_prompt
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "name": "get_current_time",
                        "toolUseId": f"tu_{self.turns}",
                    }
                }
            }
        }
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
                "metrics": {"latencyMs": 1},
            }
        }


async def test_a_runaway_tool_loop_is_capped(tmp_path, settings, alice):
    model = NeverStopsModel()
    runner = build(tmp_path, settings, model)

    result = await runner.converse(req(tools=["get_current_time"], max_tool_iterations=3), alice)
    assert model.turns == 3, f"loop ran {model.turns} turns, expected the cap"
    assert result.iterations <= 3


async def test_the_default_cap_applies_when_the_request_is_silent(tmp_path, settings, alice):
    """The dangerous case: a client that never thought about a cap."""
    model = NeverStopsModel()
    runner = build(tmp_path, settings, model)
    await runner.converse(req(tools=["get_current_time"]), alice)
    assert model.turns == AgentRunner.DEFAULT_TURNS


async def test_a_capped_turn_leaves_a_replayable_transcript(tmp_path, settings, alice):
    """Strands checks caps at the top of each iteration, so a tool requested by
    the previous turn always completes — no tool call is left without its
    result, which would break every later turn in the session."""
    model = NeverStopsModel()
    runner = build(tmp_path, settings, model)
    result = await runner.converse(req(tools=["get_current_time"], max_tool_iterations=2), alice)

    stored = await runner.read_transcript(result.session_id)
    uses, results = set(), set()
    for message in stored:
        for block in message.get("content", []):
            if "toolUse" in block:
                uses.add(block["toolUse"]["toolUseId"])
            if "toolResult" in block:
                results.add(block["toolResult"]["toolUseId"])
    assert uses, "expected tool calls in the transcript"
    assert uses == results, f"dangling tool calls: {uses ^ results}"


async def test_the_streaming_loop_is_capped_too(tmp_path, settings, alice):
    model = NeverStopsModel()
    runner = build(tmp_path, settings, model)
    _, events = await runner.converse_stream(
        req(tools=["get_current_time"], max_tool_iterations=2), alice
    )
    [_ async for _ in events]
    assert model.turns == 2


async def test_a_stalled_turn_times_out_rather_than_hanging(tmp_path, settings, alice):
    """Without a wall-clock ceiling a stalled provider holds the connection
    open for the SDK's whole timeout with nothing to show the caller."""
    import asyncio

    from morpheus.errors import ProviderTimeoutError

    class Stalls(ScriptedModel):
        async def stream(self, *a, **kw):
            await asyncio.sleep(30)
            yield {}

    settings.turn_timeout_seconds = 0.2
    runner = build(tmp_path, settings, Stalls())

    with pytest.raises(ProviderTimeoutError, match="did not finish"):
        await runner.converse(req(), alice)
