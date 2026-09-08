"""HTTP surface, wired to the fake providers."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from model_harness.api import app as app_module
from model_harness.auth.principals import hash_key
from model_harness.config import Settings
from model_harness.core.service import ConverseService
from model_harness.sessions.memory import InMemorySessionStore

from .conftest import ALICE_KEY, BOB_KEY, DISABLED_KEY, SONNET_ONLY_KEY


@pytest.fixture
def _patched_build(monkeypatch, settings, fake_anthropic, fake_openai):
    """Swap in the fake providers, sharing one store across every client."""
    store = InMemorySessionStore(ttl_seconds=0, max_turns=0)

    def build(_settings):
        return ConverseService(
            providers={"anthropic": fake_anthropic, "openai": fake_openai},
            store=store,
            settings=settings,
        )

    monkeypatch.setattr(app_module, "build_service", build)
    return build


@pytest.fixture
def client(_patched_build, settings):
    """Anonymous-mode client, for tests about routing rather than auth."""
    with TestClient(app_module.create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def authed(_patched_build, tmp_path, principal_store):
    """A client against an app with inbound auth switched on.

    Returns the TestClient plus a helper that builds an Authorization header,
    so a test can act as a specific principal.
    """
    path = tmp_path / "principals.json"
    path.write_text(
        json.dumps(
            {
                "principals": [
                    {"id": "alice", "key_sha256": hash_key(ALICE_KEY)},
                    {"id": "bob", "key_sha256": hash_key(BOB_KEY)},
                    {
                        "id": "sonnet-only",
                        "key_sha256": hash_key(SONNET_ONLY_KEY),
                        "allowed_models": ["claude-sonnet-5"],
                    },
                    {
                        "id": "retired",
                        "key_sha256": hash_key(DISABLED_KEY),
                        "disabled": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    app_settings = Settings(
        HARNESS_PRINCIPALS_FILE=path,
        HARNESS_ALLOW_ANONYMOUS=False,
        HARNESS_SESSION_TTL_SECONDS=0,
        HARNESS_SESSION_MAX_TURNS=0,
        _env_file=None,
    )
    with TestClient(app_module.create_app(app_settings)) as test_client:
        yield test_client


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def turn(model: str, text: str, session_id: str | None = None) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }
    if session_id:
        body["session_id"] = session_id
    return body


def test_health_reports_provider_status(client):
    body = client.get("/v1/health").json()
    assert set(body["providers"]) == {"anthropic", "openai"}
    assert body["providers"]["anthropic"]["available"] is True


def test_health_is_degraded_when_no_credential_was_found(monkeypatch, client):
    """`available` alone is not health.

    The Anthropic adapter reports itself available with no credential detected,
    because it resolves lazily. Reporting "ok" on that basis would mean a green
    health check in front of a service that cannot serve one request.
    """
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["usable_providers"] == ["anthropic", "openai"]

    from model_harness.providers.credentials import CredentialInfo, CredentialSource

    service = client.app.state.service
    for provider in service._providers.values():
        provider.credential = CredentialInfo(CredentialSource.UNDETECTED, "none found")

    degraded = client.get("/v1/health").json()
    assert degraded["status"] == "degraded"
    assert degraded["usable_providers"] == []


def test_models_lists_the_catalog_with_capabilities(client):
    models = {m["id"]: m for m in client.get("/v1/models").json()["models"]}

    assert models["claude-opus-5"]["provider"] == "anthropic"
    assert models["claude-opus-5"]["supports_sampling"] is False
    assert models["gpt-5"]["provider"] == "openai"
    assert models["gpt-5"]["max_effort"] == "high"
    assert models["claude-haiku-4-5"]["supports_effort"] is False


def test_converse_returns_the_turn_and_a_session_header(client):
    response = client.post(
        "/v1/converse",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["provider"] == "anthropic"
    assert body["output"]["message"]["content"][0]["text"] == "claude here"
    assert body["stop_reason"] == "end_turn"
    assert response.headers["X-Session-Id"] == body["session_id"]
    assert response.headers["X-Provider"] == "anthropic"


def test_one_session_spans_both_providers(client):
    first = client.post(
        "/v1/converse",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "remember 42"}]}],
        },
    ).json()

    second = client.post(
        "/v1/converse",
        json={
            "model": "gpt-5",
            "session_id": first["session_id"],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "recall"}]}],
        },
    ).json()

    assert second["session_id"] == first["session_id"]
    assert second["provider"] == "openai"

    transcript = client.get(
        f"/v1/sessions/{first['session_id']}", params={"include_messages": True}
    ).json()
    assert [m["content"][0]["text"] for m in transcript["messages"]] == [
        "remember 42",
        "claude here",
        "recall",
        "gpt here",
    ]
    assert transcript["turn_count"] == 2


def test_unknown_model_is_a_404_with_a_stable_code(client):
    response = client.post(
        "/v1/converse",
        json={
            "model": "gemini-ultra",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_model"


def test_unconfigured_provider_is_a_503(client, fake_openai):
    fake_openai.is_available = False
    response = client.post(
        "/v1/converse",
        json={
            "model": "gpt-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_unavailable"


def test_a_malformed_body_is_a_422(client):
    assert client.post("/v1/converse", json={"model": "claude-opus-5"}).status_code == 422


def test_an_assistant_only_request_is_a_400(client):
    response = client.post(
        "/v1/converse",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_streaming_emits_canonical_sse_events(client):
    with client.stream(
        "POST",
        "/v1/converse-stream",
        json={
            "model": "gpt-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        session_id = response.headers["X-Session-Id"]
        raw = "".join(response.iter_text())

    events = [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert [e["type"] for e in events] == [
        "message_start",
        "content_block_start",
        "content_delta",
        "content_delta",
        "content_block_stop",
        "message_stop",
    ]
    assert "".join(e["text"] for e in events if e["type"] == "content_delta").strip() == (
        "gpt here"
    )
    assert events[-1]["usage"]["output_tokens"] == 7
    assert raw.endswith("data: [DONE]\n\n")

    stored = client.get(f"/v1/sessions/{session_id}", params={"include_messages": True}).json()
    assert [m["content"][0]["text"] for m in stored["messages"]] == ["hi", "gpt here"]


def test_session_listing_and_deletion(client):
    created = client.post(
        "/v1/converse",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
    ).json()["session_id"]

    assert created in client.get("/v1/sessions").json()["session_ids"]
    assert client.delete(f"/v1/sessions/{created}").status_code == 204
    assert client.get(f"/v1/sessions/{created}").status_code == 404
    assert client.delete(f"/v1/sessions/{created}").status_code == 404


def test_session_metadata_omits_the_transcript_by_default(client):
    created = client.post(
        "/v1/converse",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
    ).json()["session_id"]

    body = client.get(f"/v1/sessions/{created}").json()
    assert body["message_count"] == 2
    assert "messages" not in body


# --- inbound authentication over HTTP -------------------------------------


def test_every_spending_endpoint_requires_a_key(authed):
    """The core inbound-auth property, asserted route by route.

    Written as a sweep rather than one case per route so a newly added route
    that forgets its principal dependency shows up here.
    """
    unauthenticated = [
        ("POST", "/v1/converse", turn("claude-opus-5", "hi")),
        ("POST", "/v1/converse-stream", turn("claude-opus-5", "hi")),
        ("GET", "/v1/models", None),
        ("GET", "/v1/sessions", None),
        ("GET", "/v1/sessions/sess_x", None),
        ("DELETE", "/v1/sessions/sess_x", None),
    ]
    for method, path, body in unauthenticated:
        response = authed.request(method, path, json=body)
        assert response.status_code == 401, f"{method} {path} did not require auth"
        assert response.json()["error"]["code"] == "authentication_required"
        assert response.headers["WWW-Authenticate"].startswith("Bearer")


def test_health_stays_open_and_leaks_no_secret(authed):
    response = authed.get("/v1/health")
    assert response.status_code == 200
    assert "test-anthropic" not in response.text
    assert "sha256" not in response.text.lower()


def test_a_valid_key_is_accepted(authed):
    response = authed.post(
        "/v1/converse", json=turn("claude-opus-5", "hi"), headers=auth(ALICE_KEY)
    )
    assert response.status_code == 200
    assert response.json()["output"]["message"]["content"][0]["text"] == "claude here"


def test_a_bad_key_is_401_without_saying_why(authed):
    for key in ("mh_wrong", DISABLED_KEY):
        response = authed.post("/v1/converse", json=turn("claude-opus-5", "hi"), headers=auth(key))
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_credential"
        assert response.json()["error"]["message"] == "The supplied API key is not valid."


def test_a_malformed_authorization_header_is_401(authed):
    for header in ({"Authorization": "Bearer"}, {"Authorization": ALICE_KEY}):
        response = authed.post("/v1/converse", json=turn("claude-opus-5", "hi"), headers=header)
        assert response.status_code == 401


def test_the_model_catalog_is_filtered_per_principal(authed):
    everything = authed.get("/v1/models", headers=auth(ALICE_KEY)).json()["models"]
    restricted = authed.get("/v1/models", headers=auth(SONNET_ONLY_KEY)).json()["models"]

    assert len(everything) > 1
    assert [m["id"] for m in restricted] == ["claude-sonnet-5"]


def test_a_forbidden_model_is_403(authed):
    response = authed.post(
        "/v1/converse", json=turn("claude-opus-5", "hi"), headers=auth(SONNET_ONLY_KEY)
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "model_not_permitted"


def test_sessions_are_isolated_between_principals(authed):
    created = authed.post(
        "/v1/converse", json=turn("claude-opus-5", "my secret is 42"), headers=auth(ALICE_KEY)
    ).json()["session_id"]

    # Bob cannot read it, cannot delete it, and cannot append to it.
    assert authed.get(f"/v1/sessions/{created}", headers=auth(BOB_KEY)).status_code == 404
    assert authed.delete(f"/v1/sessions/{created}", headers=auth(BOB_KEY)).status_code == 404

    hijack = authed.post(
        "/v1/converse",
        json=turn("claude-opus-5", "what is the secret?", created),
        headers=auth(BOB_KEY),
    )
    assert hijack.status_code == 404
    assert hijack.json()["error"]["code"] == "session_not_found"

    # Alice's transcript is intact and still hers.
    mine = authed.get(
        f"/v1/sessions/{created}", params={"include_messages": True}, headers=auth(ALICE_KEY)
    ).json()
    assert mine["owner"] == "alice"
    assert [m["content"][0]["text"] for m in mine["messages"]] == [
        "my secret is 42",
        "claude here",
    ]


def test_session_listing_shows_only_your_own(authed):
    mine = authed.post(
        "/v1/converse", json=turn("claude-opus-5", "hi"), headers=auth(ALICE_KEY)
    ).json()["session_id"]
    theirs = authed.post(
        "/v1/converse", json=turn("claude-opus-5", "hi"), headers=auth(BOB_KEY)
    ).json()["session_id"]

    assert authed.get("/v1/sessions", headers=auth(ALICE_KEY)).json()["session_ids"] == [mine]
    assert authed.get("/v1/sessions", headers=auth(BOB_KEY)).json()["session_ids"] == [theirs]


def test_streaming_requires_and_honours_auth(authed):
    with authed.stream(
        "POST", "/v1/converse-stream", json=turn("gpt-5", "hi"), headers=auth(ALICE_KEY)
    ) as response:
        assert response.status_code == 200
        session_id = response.headers["X-Session-Id"]
        "".join(response.iter_text())

    assert authed.get(f"/v1/sessions/{session_id}", headers=auth(BOB_KEY)).status_code == 404
    assert authed.get(f"/v1/sessions/{session_id}", headers=auth(ALICE_KEY)).status_code == 200


# --- error sanitization ---------------------------------------------------


def test_harness_errors_explain_themselves_and_carry_an_error_id(client):
    body = client.post("/v1/converse", json=turn("gemini-ultra", "hi")).json()["error"]
    assert body["code"] == "unknown_model"
    assert "gemini-ultra" in body["message"]
    assert body["error_id"].startswith("err_")


def test_provider_errors_are_generic_and_reference_the_error_id(client, fake_anthropic, caplog):
    """A provider failure must not relay the SDK's exception to the caller."""
    import logging

    from model_harness.errors import ProviderBadRequestError

    async def explode(_call):
        raise ProviderBadRequestError(
            "Anthropic rejected the request",
            provider="anthropic",
            detail="prompt is 900000 tokens; account org-abc123 limit exceeded",
        )

    fake_anthropic.converse = explode

    with caplog.at_level(logging.WARNING, logger="model_harness"):
        response = client.post("/v1/converse", json=turn("claude-opus-5", "hi"))

    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "provider_rejected_request"

    # The sensitive detail reaches the log, never the response.
    assert "org-abc123" not in response.text
    assert "900000" not in response.text
    assert body["error_id"] in caplog.text
    assert "org-abc123" in caplog.text


def test_a_mid_stream_failure_is_reported_as_a_sanitized_error_event(client, fake_anthropic):
    from model_harness.errors import ProviderServerError

    async def explode(_call, _sink):
        raise ProviderServerError(
            "Anthropic server error", provider="anthropic", detail="upstream trace abc"
        )
        yield  # pragma: no cover - makes this an async generator

    fake_anthropic.stream = explode

    with client.stream("POST", "/v1/converse-stream", json=turn("claude-opus-5", "hi")) as response:
        assert response.status_code == 200  # headers already sent
        raw = "".join(response.iter_text())

    events = [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "provider_server_error"
    assert events[-1]["error_id"].startswith("err_")
    assert "upstream trace abc" not in raw
    assert raw.endswith("data: [DONE]\n\n")


# --- endpoints the web client depends on ----------------------------------


def test_whoami_identifies_the_principal_without_leaking_the_key(authed):
    body = authed.get("/v1/whoami", headers=auth(ALICE_KEY)).json()
    assert body["principal_id"] == "alice"
    assert body["allowed_models"] is None  # no allowlist means all models

    restricted = authed.get("/v1/whoami", headers=auth(SONNET_ONLY_KEY)).json()
    assert restricted["principal_id"] == "sonnet-only"
    assert restricted["allowed_models"] == ["claude-sonnet-5"]

    # The credential itself, and its digest, must not appear anywhere.
    raw = authed.get("/v1/whoami", headers=auth(ALICE_KEY)).text
    assert ALICE_KEY not in raw
    assert hash_key(ALICE_KEY) not in raw


def test_whoami_requires_a_key(authed):
    assert authed.get("/v1/whoami").status_code == 401


def test_detailed_session_listing_carries_a_preview(client):
    created = client.post(
        "/v1/converse", json=turn("claude-opus-5", "Remember that my name is Ramesh")
    ).json()["session_id"]

    body = client.get("/v1/sessions", params={"detail": True}).json()
    assert [s["session_id"] for s in body["sessions"]] == [created]

    only = body["sessions"][0]
    assert only["preview"] == "Remember that my name is Ramesh"
    assert only["turn_count"] == 1
    assert only["last_provider"] == "anthropic"
    # A listing is metadata only — never the transcript.
    assert "messages" not in only


def test_a_long_preview_is_truncated_and_whitespace_collapsed(client):
    long_prompt = "word " * 60
    client.post("/v1/converse", json=turn("claude-opus-5", long_prompt))
    preview = client.get("/v1/sessions", params={"detail": True}).json()["sessions"][0]["preview"]
    assert len(preview) <= 80
    assert preview.endswith("…")
    assert "  " not in preview


def test_detailed_listing_is_scoped_to_the_principal(authed):
    authed.post("/v1/converse", json=turn("claude-opus-5", "alice one"), headers=auth(ALICE_KEY))
    authed.post("/v1/converse", json=turn("claude-opus-5", "bob one"), headers=auth(BOB_KEY))

    mine = authed.get("/v1/sessions", params={"detail": True}, headers=auth(ALICE_KEY)).json()
    assert [s["preview"] for s in mine["sessions"]] == ["alice one"]
    assert all(s["owner"] == "alice" for s in mine["sessions"])


def test_the_bare_listing_still_returns_ids(client):
    client.post("/v1/converse", json=turn("claude-opus-5", "hi"))
    body = client.get("/v1/sessions").json()
    assert "session_ids" in body and "sessions" not in body


# --- the web client is served from the same origin ------------------------


def test_the_web_client_is_served_and_needs_no_cors(client):
    """Same-origin hosting is the reason no CORS headers are configured.

    If this page ever moved to another origin, every API call from it would
    need CORS opened on a service that holds provider credentials.
    """
    page = client.get("/app/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "model-harness" in page.text

    assert client.get("/", follow_redirects=False).status_code in (307, 308)


def test_the_web_client_ships_no_credential(client):
    """The page must not embed a key — it collects one at sign-in."""
    page = client.get("/app/").text
    assert "mh_test" not in page
    assert "sk-ant" not in page
    assert "sk-proj" not in page
