"""Test fixtures: a fake provider standing in for both vendors.

The adapters' own translation logic is exercised directly in
``test_providers.py``; everything session- and routing-related is tested
against this fake so the suite needs no credentials and makes no network
calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from model_harness.auth.principals import Principal, PrincipalStore, hash_key
from model_harness.config import Settings
from model_harness.core.service import ConverseService
from model_harness.core.types import (
    Message,
    Role,
    StopReason,
    StreamEvent,
    TextBlock,
    Usage,
)
from model_harness.providers.base import ProviderCall, ProviderResult, StreamSink
from model_harness.providers.credentials import CredentialInfo, CredentialSource
from model_harness.sessions.memory import InMemorySessionStore


@dataclass
class FakeProvider:
    """Records every call so a test can assert on what the provider was sent."""

    name: str
    is_available: bool = True
    reply: str = "ok"
    calls: list[ProviderCall] = field(default_factory=list)
    native_payloads: list[Any] = field(default_factory=list)
    credential: CredentialInfo = field(
        default_factory=lambda: CredentialInfo(CredentialSource.API_KEY, "test")
    )

    def available(self) -> bool:
        return self.is_available

    def _result(self, call: ProviderCall) -> ProviderResult:
        native = {"role": "assistant", "content": self.reply, "_provider": self.name}
        self.native_payloads.append(native)
        return ProviderResult(
            message=Message(
                role=Role.ASSISTANT,
                content=[TextBlock(text=self.reply)],
                provider=self.name,
                native=native,
            ),
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=11, output_tokens=7),
            native_model=f"{call.spec.native_id}-native",
        )

    async def converse(self, call: ProviderCall) -> ProviderResult:
        self.calls.append(call)
        return self._result(call)

    async def stream(self, call: ProviderCall, sink: StreamSink) -> AsyncIterator[StreamEvent]:
        self.calls.append(call)
        yield StreamEvent(type="message_start", provider=self.name, model=call.spec.id)
        yield StreamEvent(type="content_block_start", index=0, block_type="text")
        for chunk in self.reply.split(" "):
            yield StreamEvent(type="content_delta", index=0, text=chunk + " ")
        yield StreamEvent(type="content_block_stop", index=0)

        result = self._result(call)
        sink.result = result
        yield StreamEvent(
            type="message_stop",
            provider=self.name,
            model=call.spec.id,
            native_model=result.native_model,
            stop_reason=result.stop_reason,
            usage=result.usage,
        )

    # last transcript the provider was asked to serve
    @property
    def last_transcript(self) -> list[Message]:
        return self.calls[-1].messages


@pytest.fixture
def settings() -> Settings:
    return Settings(
        ANTHROPIC_API_KEY="test-anthropic",
        OPENAI_API_KEY="test-openai",
        HARNESS_SESSION_TTL_SECONDS=0,
        HARNESS_SESSION_MAX_TURNS=0,
        HARNESS_ALLOW_ANONYMOUS=True,
        _env_file=None,
    )


# --- inbound principals ---------------------------------------------------

ALICE_KEY = "mh_test_alice_key"
BOB_KEY = "mh_test_bob_key"
SONNET_ONLY_KEY = "mh_test_sonnet_only_key"
DISABLED_KEY = "mh_test_disabled_key"


@pytest.fixture
def alice() -> Principal:
    return Principal(id="alice", key_sha256=hash_key(ALICE_KEY))


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


@pytest.fixture
def fake_anthropic() -> FakeProvider:
    return FakeProvider(name="anthropic", reply="claude here")


@pytest.fixture
def fake_openai() -> FakeProvider:
    return FakeProvider(name="openai", reply="gpt here")


@pytest.fixture
def service(settings, fake_anthropic, fake_openai) -> ConverseService:
    return ConverseService(
        providers={"anthropic": fake_anthropic, "openai": fake_openai},
        store=InMemorySessionStore(ttl_seconds=0, max_turns=0),
        settings=settings,
    )
