"""HTTP surface, wired to a scripted Strands model."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from model_harness.api import app as app_module
from model_harness.api import deps
from model_harness.auth.principals import hash_key
from model_harness.config import Settings

from .conftest import ALICE_KEY, BOB_KEY, DISABLED_KEY, SONNET_ONLY_KEY, ScriptedModel


@pytest.fixture
def script() -> list[dict]:
    """Mutable script the test can rewrite before each request."""
    return [{"text": "hello from the model"}]


@pytest.fixture
def _wire(monkeypatch, script):
    """Point the app at a scripted model instead of a real provider."""
    model = ScriptedModel(script)
    real = deps.build_runner

    def build(settings, model_override=None, **kwargs):
        return real(settings, model_override=model, **kwargs)

    monkeypatch.setattr(app_module, "build_runner", build)
    return model


@pytest.fixture
def client(_wire, settings):
    with TestClient(app_module.create_app(settings)) as c:
        yield c


@pytest.fixture
def authed(_wire, tmp_path, settings):
    """A client with inbound auth switched on."""
    path = tmp_path / "principals.json"
    path.write_text(
        json.dumps(
            {
                "principals": [
                    {
                        "id": "alice",
                        "key_sha256": hash_key(ALICE_KEY),
                        "allowed_tools": ["*"],
                    },
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
        ANTHROPIC_API_KEY="test-anthropic",
        OPENAI_API_KEY="test-openai",
        HARNESS_PRINCIPALS_FILE=path,
        HARNESS_ALLOW_ANONYMOUS=False,
        HARNESS_SESSION_DIR=str(tmp_path / "authed-sessions"),
        _env_file=None,
    )
    with TestClient(app_module.create_app(app_settings)) as c:
        yield c


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def turn(model="claude-opus-5", text="hi", session_id=None, **extra) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        **extra,
    }
    if session_id:
        body["session_id"] = session_id
    return body


# --- catalog and health ---------------------------------------------------


def test_health_is_open_and_reports_credential_sources(client):
    body = client.get("/v1/health").json()
    assert set(body["providers"]) == {"anthropic", "openai"}
    assert body["providers"]["anthropic"]["credential_source"] == "api_key"
    assert "test-anthropic" not in client.get("/v1/health").text


def test_models_are_filtered_to_the_principal(authed):
    everything = authed.get("/v1/models", headers=auth(ALICE_KEY)).json()["models"]
    restricted = authed.get("/v1/models", headers=auth(SONNET_ONLY_KEY)).json()["models"]
    assert len(everything) > 1
    assert [m["id"] for m in restricted] == ["claude-sonnet-5"]


def test_tools_lists_permission_per_principal(authed):
    mine = {t["name"]: t for t in authed.get("/v1/tools", headers=auth(ALICE_KEY)).json()["tools"]}
    theirs = {t["name"]: t for t in authed.get("/v1/tools", headers=auth(BOB_KEY)).json()["tools"]}

    assert mine["http_request"]["permitted"] is True
    # bob has no allowed_tools, so dangerous tools are denied but still listed
    assert theirs["http_request"]["permitted"] is False
    assert theirs["http_request"]["dangerous"] is True
    assert theirs["get_current_time"]["permitted"] is True


def test_whoami_leaks_no_secret(authed):
    body = authed.get("/v1/whoami", headers=auth(ALICE_KEY)).json()
    assert body["principal_id"] == "alice"
    raw = authed.get("/v1/whoami", headers=auth(ALICE_KEY)).text
    assert ALICE_KEY not in raw and hash_key(ALICE_KEY) not in raw


# --- converse -------------------------------------------------------------


def test_converse_returns_the_turn_and_headers(client):
    response = client.post("/v1/converse", json=turn())
    assert response.status_code == 200
    body = response.json()

    assert body["output"]["message"]["content"][0]["text"] == "hello from the model"
    assert body["provider"] == "anthropic"
    assert body["usage"]["input_tokens"] == 11
    assert response.headers["X-Session-Id"] == body["session_id"]


def test_a_session_continues_across_requests(client):
    first = client.post("/v1/converse", json=turn(text="remember 42")).json()
    second = client.post(
        "/v1/converse", json=turn(text="recall", session_id=first["session_id"])
    ).json()
    assert second["session_id"] == first["session_id"]

    stored = client.get(
        f"/v1/sessions/{first['session_id']}", params={"include_messages": True}
    ).json()
    texts = [
        c.get("text", "")
        for m in stored["messages"]
        for c in m.get("content", [])
        if isinstance(c, dict)
    ]
    assert any("remember 42" in t for t in texts)
    assert stored["turn_count"] == 2


def test_unknown_model_and_forbidden_model(client, authed):
    assert client.post("/v1/converse", json=turn(model="gemini-ultra")).status_code == 404
    forbidden = authed.post(
        "/v1/converse", json=turn(model="claude-opus-5"), headers=auth(SONNET_ONLY_KEY)
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "model_not_permitted"


def test_a_dangerous_tool_is_denied_by_default(authed):
    response = authed.post("/v1/converse", json=turn(tools=["http_request"]), headers=auth(BOB_KEY))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "tool_not_permitted"


def test_a_malformed_body_is_422(client):
    assert client.post("/v1/converse", json={"model": "claude-opus-5"}).status_code == 422


# --- inbound auth ---------------------------------------------------------


def test_every_spending_endpoint_requires_a_key(authed):
    for method, path, body in [
        ("POST", "/v1/converse", turn()),
        ("POST", "/v1/converse-stream", turn()),
        ("GET", "/v1/models", None),
        ("GET", "/v1/tools", None),
        ("GET", "/v1/whoami", None),
        ("GET", "/v1/sessions", None),
        ("GET", "/v1/sessions/sess_x", None),
        ("DELETE", "/v1/sessions/sess_x", None),
    ]:
        r = authed.request(method, path, json=body)
        assert r.status_code == 401, f"{method} {path} did not require auth"
        assert r.headers["WWW-Authenticate"].startswith("Bearer")


def test_a_bad_or_disabled_key_is_indistinguishable(authed):
    messages = set()
    for key in ("mh_wrong", DISABLED_KEY):
        r = authed.post("/v1/converse", json=turn(), headers=auth(key))
        assert r.status_code == 401
        messages.add(r.json()["error"]["message"])
    assert len(messages) == 1


def test_sessions_are_isolated_between_principals(authed):
    created = authed.post(
        "/v1/converse", json=turn(text="my secret is 42"), headers=auth(ALICE_KEY)
    ).json()["session_id"]

    assert authed.get(f"/v1/sessions/{created}", headers=auth(BOB_KEY)).status_code == 404
    assert authed.delete(f"/v1/sessions/{created}", headers=auth(BOB_KEY)).status_code == 404

    hijack = authed.post(
        "/v1/converse", json=turn(text="what is it?", session_id=created), headers=auth(BOB_KEY)
    )
    assert hijack.status_code == 404
    assert hijack.json()["error"]["code"] == "session_not_found"

    mine = authed.get(f"/v1/sessions/{created}", headers=auth(ALICE_KEY)).json()
    assert mine["owner"] == "alice"


def test_session_listing_shows_only_your_own(authed):
    mine = authed.post("/v1/converse", json=turn(), headers=auth(ALICE_KEY)).json()["session_id"]
    theirs = authed.post("/v1/converse", json=turn(), headers=auth(BOB_KEY)).json()["session_id"]
    assert authed.get("/v1/sessions", headers=auth(ALICE_KEY)).json()["session_ids"] == [mine]
    assert authed.get("/v1/sessions", headers=auth(BOB_KEY)).json()["session_ids"] == [theirs]


def test_deleting_a_session_removes_its_transcript(client):
    created = client.post("/v1/converse", json=turn(text="hello")).json()["session_id"]
    assert client.delete(f"/v1/sessions/{created}").status_code == 204
    assert client.get(f"/v1/sessions/{created}").status_code == 404
    assert client.delete(f"/v1/sessions/{created}").status_code == 404


def test_detailed_listing_carries_a_preview(client):
    client.post("/v1/converse", json=turn(text="Remember my name is Ramesh"))
    only = client.get("/v1/sessions", params={"detail": True}).json()["sessions"][0]
    assert only["preview"] == "Remember my name is Ramesh"
    assert only["turn_count"] == 1
    assert "messages" not in only


# --- streaming, tools now allowed ----------------------------------------


def sse_events(raw: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_streaming_emits_canonical_events(client):
    with client.stream("POST", "/v1/converse-stream", json=turn()) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        session_id = r.headers["X-Session-Id"]
        raw = "".join(r.iter_text())

    events = sse_events(raw)
    assert [e["type"] for e in events] == [
        "message_start",
        "content_block_start",
        "content_delta",
        "content_block_stop",
        "message_stop",
    ]
    assert events[0]["session_id"] == session_id
    assert events[-1]["usage"]["output_tokens"] == 7
    assert raw.endswith("data: [DONE]\n\n")


def test_streaming_with_tools_is_no_longer_refused(client, script):
    """This used to be a 400: the old implementation piped one provider
    response through, and a tool loop is several. Strands streams the whole
    loop, so the restriction was ours, not the problem's."""
    script[:] = [
        {"tool": ("get_current_time", {"timezone": "UTC"}, "tu_1")},
        {"text": "It is 09:00."},
    ]
    with client.stream("POST", "/v1/converse-stream", json=turn(tools=["get_current_time"])) as r:
        assert r.status_code == 200
        raw = "".join(r.iter_text())

    events = sse_events(raw)
    kinds = [e["type"] for e in events]
    assert "tool_start" in kinds
    assert "tool_end" in kinds
    # The tool lifecycle must precede the text, or the client has nothing to
    # show during the silence while the tool runs.
    assert kinds.index("tool_start") < kinds.index("content_delta")

    start = next(e for e in events if e["type"] == "tool_start")
    end = next(e for e in events if e["type"] == "tool_end")
    assert start["tool_name"] == "get_current_time"
    assert start["tool_use_id"] == end["tool_use_id"]
    assert end["is_error"] is False
    assert events[-1]["iterations"] == 2


def test_tool_output_is_withheld_from_the_stream_by_default(client, script):
    script[:] = [
        {"tool": ("get_current_time", {"timezone": "UTC"}, "tu_1")},
        {"text": "done"},
    ]
    with client.stream("POST", "/v1/converse-stream", json=turn(tools=["get_current_time"])) as r:
        raw = "".join(r.iter_text())
    end = next(e for e in sse_events(raw) if e["type"] == "tool_end")
    assert "tool_output" not in end


def test_tool_output_is_included_when_asked_for(client, script):
    script[:] = [
        {"tool": ("get_current_time", {"timezone": "UTC"}, "tu_1")},
        {"text": "done"},
    ]
    with client.stream(
        "POST",
        "/v1/converse-stream",
        json=turn(tools=["get_current_time"], stream_tool_output=True),
    ) as r:
        raw = "".join(r.iter_text())
    end = next(e for e in sse_events(raw) if e["type"] == "tool_end")
    assert "iso8601" in end["tool_output"]


def test_a_streamed_turn_is_persisted(client):
    with client.stream("POST", "/v1/converse-stream", json=turn(text="stream me")) as r:
        session_id = r.headers["X-Session-Id"]
        "".join(r.iter_text())

    stored = client.get(f"/v1/sessions/{session_id}", params={"include_messages": True}).json()
    assert stored["turn_count"] == 1
    assert stored["messages"]


def test_streaming_respects_session_ownership(authed):
    with authed.stream("POST", "/v1/converse-stream", json=turn(), headers=auth(ALICE_KEY)) as r:
        session_id = r.headers["X-Session-Id"]
        "".join(r.iter_text())

    assert authed.get(f"/v1/sessions/{session_id}", headers=auth(BOB_KEY)).status_code == 404
    assert authed.get(f"/v1/sessions/{session_id}", headers=auth(ALICE_KEY)).status_code == 200


# --- web client -----------------------------------------------------------


def test_the_web_client_is_served_same_origin(client):
    page = client.get("/app/")
    assert page.status_code == 200
    assert "model-harness" in page.text
    assert client.get("/", follow_redirects=False).status_code in (307, 308)
    for secret in ("mh_test", "sk-ant", "sk-proj"):
        assert secret not in page.text


# --- the API must never put non-JSON on the wire -------------------------


def test_an_unhandled_exception_still_returns_json(_wire, settings, monkeypatch):
    """A 500 used to arrive as the plain text "Internal Server Error", so any
    client parsing JSON failed with a parse error instead of showing the
    problem. That is how a provider misconfiguration surfaced in the web
    client as `Unexpected token 'I'`."""
    from model_harness.harness.runner import AgentRunner

    async def boom(self, request, principal):
        raise KeyError("max_tokens")

    monkeypatch.setattr(AgentRunner, "converse", boom)

    # TestClient re-raises server exceptions by default, which would bypass
    # the handler under test; uvicorn does not.
    with TestClient(app_module.create_app(settings), raise_server_exceptions=False) as raw:
        response = raw.post("/v1/converse", json=turn())
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()["error"]
    assert body["code"] == "internal_error"
    assert body["error_id"].startswith("err_")
    # The exception detail is logged, never returned.
    assert "max_tokens" not in response.text


def test_every_catalog_model_can_actually_be_constructed(settings):
    """The bug that caused the 500: provider config is **flat kwargs**, so
    nesting it in `model_config=` landed in an unknown key and the model then
    raised KeyError on first use. Nothing caught it because every other test
    injects a scripted model."""
    import warnings

    from model_harness.core.registry import known_ids, resolve
    from model_harness.harness.models import ModelFactory

    factory = ModelFactory(settings)
    for model_id in known_ids():
        spec = resolve(model_id)
        with warnings.catch_warnings():
            # Strands warns rather than raises on an unknown config key, which
            # is how the original bug stayed silent until request time.
            warnings.simplefilter("error")
            model = factory.build(spec, max_tokens=1024)
        config = model.get_config()
        assert config["model_id"] == spec.native_id, model_id


# --- MCP through the HTTP surface ----------------------------------------


@pytest.fixture
def mcp_client(_wire, tmp_path):
    """A client with one real MCP server, and a principal granted its tools.

    Authenticated rather than anonymous on purpose: MCP tools are dangerous by
    definition, and the harness never grants a dangerous tool without an
    explicit grant — not even in anonymous mode. Using them therefore requires
    a principals file, which is the security model working as intended.
    """
    import json as _json
    import sys
    from pathlib import Path as _Path

    server = _Path(__file__).parent / "fixtures" / "tiny_mcp_server.py"
    config = tmp_path / "mcp.json"
    config.write_text(
        _json.dumps(
            {
                "mcpServers": {
                    "tiny": {
                        "command": sys.executable,
                        "args": [str(server)],
                        "prefix": "tiny",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    principals = tmp_path / "principals.json"
    principals.write_text(
        _json.dumps(
            {
                "principals": [
                    # A glob grant: one entry covers the whole server.
                    {
                        "id": "alice",
                        "key_sha256": hash_key(ALICE_KEY),
                        "allowed_tools": ["tiny_*"],
                    },
                    {"id": "bob", "key_sha256": hash_key(BOB_KEY)},
                ]
            }
        ),
        encoding="utf-8",
    )

    app_settings = Settings(
        ANTHROPIC_API_KEY="test-anthropic",
        HARNESS_PRINCIPALS_FILE=principals,
        HARNESS_ALLOW_ANONYMOUS=False,
        HARNESS_SESSION_DIR=str(tmp_path / "mcp-sessions"),
        HARNESS_MCP_CONFIG=str(config),
        _env_file=None,
    )
    with TestClient(app_module.create_app(app_settings)) as c:
        yield c


def test_configured_mcp_tools_appear_in_the_catalog(mcp_client):
    tools = {
        t["name"]: t for t in mcp_client.get("/v1/tools", headers=auth(ALICE_KEY)).json()["tools"]
    }

    assert "tiny_echo" in tools and "tiny_add" in tools
    assert tools["tiny_echo"]["source"] == "mcp:tiny"
    assert tools["tiny_echo"]["dangerous"] is True
    assert tools["tiny_echo"]["description"] == "Echo the text back."
    assert tools["tiny_echo"]["permitted"] is True
    # The builtins are still there alongside them.
    assert tools["get_current_time"]["source"] == "builtin"


def test_a_glob_grant_covers_a_whole_server(mcp_client):
    """Adding a server should not mean re-enumerating every principal."""
    mine = {
        t["name"]: t["permitted"]
        for t in mcp_client.get("/v1/tools", headers=auth(ALICE_KEY)).json()["tools"]
    }
    theirs = {
        t["name"]: t["permitted"]
        for t in mcp_client.get("/v1/tools", headers=auth(BOB_KEY)).json()["tools"]
    }

    assert mine["tiny_echo"] and mine["tiny_add"]
    # bob has no allowed_tools, so every MCP tool is denied but still listed —
    # he can see what to ask an operator for.
    assert not theirs["tiny_echo"] and not theirs["tiny_add"]
    assert theirs["get_current_time"] is True


def test_an_ungranted_mcp_tool_is_403(mcp_client):
    response = mcp_client.post(
        "/v1/converse", json=turn(tools=["tiny_echo"]), headers=auth(BOB_KEY)
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "tool_not_permitted"
    assert "glob" in response.json()["error"]["message"]


def test_an_mcp_tool_is_offered_to_the_model_with_its_schema(mcp_client, _wire):
    response = mcp_client.post(
        "/v1/converse", json=turn(tools=["tiny_echo"]), headers=auth(ALICE_KEY)
    )
    assert response.status_code == 200
    assert [s["name"] for s in _wire.last_tool_specs] == ["tiny_echo"]
    assert _wire.last_tool_specs[0]["inputSchema"]


def test_a_model_can_actually_call_an_mcp_tool(mcp_client, script):
    """End to end: the model asks, the harness routes over stdio to a real MCP
    server, and the real result comes back."""
    script[:] = [
        {"tool": ("tiny_echo", {"text": "hello mcp"}, "tu_1")},
        {"text": "The server said hello."},
    ]
    body = mcp_client.post(
        "/v1/converse", json=turn(tools=["tiny_echo"]), headers=auth(ALICE_KEY)
    ).json()

    assert body["iterations"] == 2
    assert [c["name"] for c in body["tool_calls"]] == ["tiny_echo"]
    assert body["tool_calls"][0]["is_error"] is False


def test_an_mcp_tool_result_reaches_the_stream(mcp_client, script):
    script[:] = [
        {"tool": ("tiny_echo", {"text": "streamed"}, "tu_1")},
        {"text": "done"},
    ]
    with mcp_client.stream(
        "POST",
        "/v1/converse-stream",
        json=turn(tools=["tiny_echo"], stream_tool_output=True),
        headers=auth(ALICE_KEY),
    ) as r:
        raw = "".join(r.iter_text())

    end = next(e for e in sse_events(raw) if e["type"] == "tool_end")
    assert end["tool_name"] == "tiny_echo"
    assert "echo: streamed" in end["tool_output"]


def test_an_unconfigured_mcp_tool_is_unknown(mcp_client):
    response = mcp_client.post(
        "/v1/converse", json=turn(tools=["github_create_issue"]), headers=auth(ALICE_KEY)
    )
    assert response.status_code == 400
    assert "Unknown tool" in response.json()["error"]["message"]
