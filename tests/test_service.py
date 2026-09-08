"""Context-preservation behaviour: the point of the whole service."""

from __future__ import annotations

import pytest

from model_harness.core.types import (
    ConverseRequest,
    Effort,
    InferenceConfig,
    Message,
    ReasoningBlock,
    Role,
    SystemBlock,
    TextBlock,
)
from model_harness.errors import (
    InvalidRequestError,
    ModelNotPermittedError,
    ProviderUnavailableError,
    SessionNotFoundError,
    UnknownModelError,
)


def user(text: str) -> Message:
    return Message(role=Role.USER, content=[TextBlock(text=text)])


def req(model: str, text: str, session_id: str | None = None, **kw) -> ConverseRequest:
    return ConverseRequest(model=model, session_id=session_id, messages=[user(text)], **kw)


async def test_new_session_is_created_and_returned(service, alice):
    result = await service.converse(req("claude-opus-5", "hello"), alice)
    assert result.session_id.startswith("sess_")
    assert result.provider == "anthropic"
    assert result.output.message.text() == "claude here"
    assert result.usage.total_tokens == 18


async def test_history_is_replayed_on_the_next_turn(service, alice, fake_anthropic):
    first = await service.converse(req("claude-opus-5", "my name is Ramesh"), alice)
    await service.converse(req("claude-opus-5", "what is my name?", first.session_id), alice)

    transcript = fake_anthropic.last_transcript
    assert [m.role for m in transcript] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert transcript[0].text() == "my name is Ramesh"
    assert transcript[2].text() == "what is my name?"


async def test_same_provider_replay_uses_the_native_payload(service, alice, fake_anthropic):
    first = await service.converse(req("claude-opus-5", "one"), alice)
    await service.converse(req("claude-opus-5", "two", first.session_id), alice)

    replayed = fake_anthropic.last_transcript[1]
    assert replayed.provider == "anthropic"
    assert replayed.native == fake_anthropic.native_payloads[0]


async def test_switching_provider_mid_session_keeps_the_transcript(service, alice, fake_openai):
    first = await service.converse(req("claude-opus-5", "my name is Ramesh"), alice)
    second = await service.converse(req("gpt-5", "what is my name?", first.session_id), alice)

    assert second.session_id == first.session_id
    assert second.provider == "openai"

    transcript = fake_openai.last_transcript
    assert [m.text() for m in transcript] == [
        "my name is Ramesh",
        "claude here",
        "what is my name?",
    ]
    # The Anthropic assistant turn is still marked as Anthropic-native, which is
    # exactly what tells the OpenAI adapter to fall back to canonical text.
    assert transcript[1].provider == "anthropic"
    assert any("provider switched anthropic -> openai" in a for a in second.adjustments)


async def test_switching_back_and_forth_keeps_growing_one_transcript(service, alice):
    first = await service.converse(req("claude-opus-5", "a"), alice)
    sid = first.session_id
    await service.converse(req("gpt-5", "b", sid), alice)
    await service.converse(req("claude-opus-5", "c", sid), alice)

    session = await service.get_session(sid, alice)
    assert [m.text() for m in session.messages] == [
        "a",
        "claude here",
        "b",
        "gpt here",
        "c",
        "claude here",
    ]
    assert session.turn_count == 3
    assert session.last_provider == "anthropic"


async def test_system_prompt_is_remembered_for_later_turns(service, alice, fake_anthropic):
    first = await service.converse(
        ConverseRequest(
            model="claude-opus-5",
            messages=[user("hi")],
            system=[SystemBlock(text="You are terse.")],
        ),
        alice,
    )
    await service.converse(req("claude-opus-5", "again", first.session_id), alice)

    assert fake_anthropic.calls[-1].system == [SystemBlock(text="You are terse.")]


async def test_changing_the_system_prompt_mid_session_is_reported(service, alice):
    first = await service.converse(
        ConverseRequest(
            model="claude-opus-5",
            messages=[user("hi")],
            system=[SystemBlock(text="A")],
        ),
        alice,
    )
    second = await service.converse(
        ConverseRequest(
            model="claude-opus-5",
            session_id=first.session_id,
            messages=[user("hi again")],
            system=[SystemBlock(text="B")],
        ),
        alice,
    )
    assert any("system prompt changed" in a for a in second.adjustments)


async def test_max_tokens_is_capped_to_the_model_ceiling(service, alice, fake_openai):
    result = await service.converse(
        ConverseRequest(
            model="gpt-4o",
            messages=[user("hi")],
            inference_config=InferenceConfig(max_tokens=100_000),
        ),
        alice,
    )
    assert fake_openai.calls[-1].max_tokens == 16_384
    assert any("lowered to 16384" in a for a in result.adjustments)


async def test_default_max_tokens_differs_between_buffered_and_streaming(
    service, alice, settings, fake_anthropic
):
    await service.converse(req("claude-opus-5", "hi"), alice)
    assert fake_anthropic.calls[-1].max_tokens == settings.default_max_tokens

    _, events = await service.converse_stream(req("claude-opus-5", "hi"), alice)
    [_ async for _ in events]
    assert fake_anthropic.calls[-1].max_tokens == settings.default_max_tokens_streaming


async def test_effort_is_forwarded_to_the_provider(service, alice, fake_anthropic):
    await service.converse(req("claude-opus-5", "hi", effort=Effort.XHIGH), alice)
    assert fake_anthropic.calls[-1].effort is Effort.XHIGH


async def test_unknown_model_is_a_404(service, alice):
    with pytest.raises(UnknownModelError):
        await service.converse(req("mistral-large", "hi"), alice)


async def test_model_whose_provider_lacks_a_credential_is_a_503(service, alice, fake_openai):
    fake_openai.is_available = False
    with pytest.raises(ProviderUnavailableError):
        await service.converse(req("gpt-5", "hi"), alice)


async def test_last_message_must_be_from_the_user(service, alice):
    bad = ConverseRequest(
        model="claude-opus-5",
        messages=[Message(role=Role.ASSISTANT, content=[TextBlock(text="hi")])],
    )
    with pytest.raises(InvalidRequestError):
        await service.converse(bad, alice)


async def test_clients_cannot_supply_reasoning_blocks(service, alice):
    bad = ConverseRequest(
        model="claude-opus-5",
        messages=[Message(role=Role.USER, content=[ReasoningBlock(text="peek")])],
    )
    with pytest.raises(InvalidRequestError):
        await service.converse(bad, alice)


async def test_reasoning_and_native_state_are_not_exposed_over_the_wire(service, alice):
    result = await service.converse(req("claude-opus-5", "hi"), alice)
    dumped = result.output.message.model_dump()
    assert "native" not in dumped
    assert "provider" not in dumped


async def test_unknown_session_id_is_adopted_rather_than_rejected(service, alice):
    result = await service.converse(req("claude-opus-5", "hi", "sess_does_not_exist"), alice)
    assert result.session_id == "sess_does_not_exist"
    assert (await service.get_session("sess_does_not_exist", alice)) is not None


async def test_streaming_persists_the_turn(service, alice):
    session_id, events = await service.converse_stream(req("claude-opus-5", "hello"), alice)
    types = [e.type async for e in events]

    assert types[0] == "message_start"
    assert types[-1] == "message_stop"

    session = await service.get_session(session_id, alice)
    assert [m.text() for m in session.messages] == ["hello", "claude here"]


async def test_deleting_a_session_forgets_the_context(service, alice):
    first = await service.converse(req("claude-opus-5", "hi"), alice)
    assert await service.delete_session(first.session_id, alice) is True
    assert await service.get_session(first.session_id, alice) is None
    assert await service.delete_session(first.session_id, alice) is False


# --- authorization: model allowlist and session ownership -----------------


async def test_a_principal_can_use_a_model_on_its_allowlist(service, sonnet_only):
    result = await service.converse(req("claude-sonnet-5", "hi"), sonnet_only)
    assert result.model == "claude-sonnet-5"


async def test_a_principal_cannot_use_a_model_off_its_allowlist(service, sonnet_only):
    with pytest.raises(ModelNotPermittedError) as caught:
        await service.converse(req("claude-opus-5", "hi"), sonnet_only)
    assert "sonnet-only" in caught.value.message


async def test_an_alias_cannot_bypass_the_allowlist(service, sonnet_only):
    """`opus` resolves to claude-opus-5, which this principal may not use."""
    with pytest.raises(ModelNotPermittedError):
        await service.converse(req("opus", "hi"), sonnet_only)


async def test_a_principal_token_ceiling_is_applied_after_the_model_ceiling(
    service, sonnet_only, fake_anthropic
):
    result = await service.converse(
        ConverseRequest(
            model="claude-sonnet-5",
            messages=[user("hi")],
            inference_config=InferenceConfig(max_tokens=90_000),
        ),
        sonnet_only,
    )
    assert fake_anthropic.calls[-1].max_tokens == 500
    assert any("ceiling for principal 'sonnet-only'" in a for a in result.adjustments)


async def test_a_session_is_owned_by_its_creator(service, alice):
    result = await service.converse(req("claude-opus-5", "hi"), alice)
    session = await service.get_session(result.session_id, alice)
    assert session.owner == "alice"


async def test_another_principal_cannot_read_someone_elses_session(service, alice, bob):
    first = await service.converse(req("claude-opus-5", "my secret is 42"), alice)

    # Reads as absent, not forbidden — a 403 would confirm the id exists.
    assert await service.get_session(first.session_id, bob) is None
    assert await service.delete_session(first.session_id, bob) is False

    # And it is genuinely still there for its owner.
    assert await service.get_session(first.session_id, alice) is not None


async def test_another_principal_cannot_write_into_someone_elses_session(service, alice, bob):
    first = await service.converse(req("claude-opus-5", "my secret is 42"), alice)

    with pytest.raises(SessionNotFoundError):
        await service.converse(req("claude-opus-5", "what is the secret?", first.session_id), bob)

    # Alice's transcript is untouched by the attempt.
    session = await service.get_session(first.session_id, alice)
    assert [m.text() for m in session.messages] == ["my secret is 42", "claude here"]


async def test_session_listing_is_scoped_to_the_principal(service, alice, bob):
    mine = await service.converse(req("claude-opus-5", "hi"), alice)
    theirs = await service.converse(req("claude-opus-5", "hi"), bob)

    assert await service.list_sessions(alice) == [mine.session_id]
    assert await service.list_sessions(bob) == [theirs.session_id]


async def test_two_principals_may_hold_the_same_session_id_without_collision(service, alice, bob):
    """Only an *unknown* id is adopted; a live one belonging to another
    principal is refused, so adoption cannot be used to hijack an id."""
    await service.converse(req("claude-opus-5", "alice turn", "sess_shared"), alice)

    with pytest.raises(SessionNotFoundError):
        await service.converse(req("claude-opus-5", "bob turn", "sess_shared"), bob)


async def test_streaming_respects_session_ownership(service, alice, bob):
    first = await service.converse(req("claude-opus-5", "hi"), alice)
    with pytest.raises(SessionNotFoundError):
        await service.converse_stream(req("claude-opus-5", "hi", first.session_id), bob)


async def test_a_failed_first_turn_leaves_no_empty_session(service, alice, fake_anthropic):
    """A session is allocated before the provider is called, so a failure must
    clean it up rather than litter the caller's session list."""
    from model_harness.errors import ProviderServerError

    async def explode(_call):
        raise ProviderServerError("boom", provider="anthropic")

    fake_anthropic.converse = explode

    with pytest.raises(ProviderServerError):
        await service.converse(req("claude-opus-5", "hi"), alice)

    assert await service.list_sessions(alice) == []


async def test_a_failed_turn_does_not_destroy_an_established_session(
    service, alice, fake_anthropic
):
    first = await service.converse(req("claude-opus-5", "hello"), alice)

    from model_harness.errors import ProviderServerError

    async def explode(_call):
        raise ProviderServerError("boom", provider="anthropic")

    fake_anthropic.converse = explode
    with pytest.raises(ProviderServerError):
        await service.converse(req("claude-opus-5", "again", first.session_id), alice)

    session = await service.get_session(first.session_id, alice)
    assert [m.text() for m in session.messages] == ["hello", "claude here"]


async def test_an_adopted_id_is_not_kept_when_the_first_turn_fails(service, alice, fake_anthropic):
    """Guards the exact clutter seen in manual testing: a client sending a
    stale or bogus id (even the string "null") should not create a permanent
    empty session when the call fails."""
    from model_harness.errors import ProviderServerError

    async def explode(_call):
        raise ProviderServerError("boom", provider="anthropic")

    fake_anthropic.converse = explode
    with pytest.raises(ProviderServerError):
        await service.converse(req("claude-opus-5", "hi", "null"), alice)

    assert await service.get_session("null", alice) is None
