"""MCP configuration, against a real stdio server.

Mocking MCP would test nothing that breaks in practice: the failure modes are
transport, tool-name prefixing, credential handling and lifecycle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from morpheus.auth.principals import Principal, hash_key
from morpheus.errors import ToolNotPermittedError
from morpheus.harness.mcp import McpRegistry
from morpheus.harness.tools import ToolCatalog

SERVER = Path(__file__).parent / "fixtures" / "tiny_mcp_server.py"


def write_config(tmp_path: Path, servers: dict) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def tiny(prefix: str = "tiny", **extra) -> dict:
    return {"command": sys.executable, "args": [str(SERVER)], "prefix": prefix, **extra}


@pytest.fixture
def registry(tmp_path):
    """A started registry with one real MCP server, stopped after the test."""
    reg = McpRegistry(write_config(tmp_path, {"tiny": tiny()}))
    reg.start()
    yield reg
    reg.stop()


# --- discovery ------------------------------------------------------------


def test_tools_are_discovered_from_a_real_server(registry):
    assert registry.errors == []
    assert sorted(registry.tools) == ["tiny_add", "tiny_echo"]
    assert registry.server_names == ["tiny"]


def test_tool_names_are_prefixed_by_server(tmp_path):
    """Prefixing is what stops two servers colliding on a common name like
    `search`, so the name shape is load-bearing for grants."""
    reg = McpRegistry(write_config(tmp_path, {"tiny": tiny(prefix="gh")}))
    reg.start()
    try:
        assert sorted(reg.tools) == ["gh_add", "gh_echo"]
    finally:
        reg.stop()


def test_schemas_and_descriptions_come_from_the_server(registry):
    described = {t["name"]: t for t in registry.describe()}
    assert described["tiny_echo"]["description"] == "Echo the text back."
    assert described["tiny_echo"]["source"] == "mcp:tiny"
    schema = json.dumps(described["tiny_add"]["input_schema"])
    assert '"a"' in schema and '"b"' in schema


def test_every_mcp_tool_is_dangerous(registry):
    """It runs code we did not write, against a system we do not control,
    with credentials this service holds."""
    assert all(t["dangerous"] for t in registry.describe())


# --- configuration --------------------------------------------------------


def test_a_disabled_server_is_skipped(tmp_path):
    path = write_config(
        tmp_path,
        {"tiny": tiny(), "off": {"command": "false", "args": [], "disabled": True}},
    )
    reg = McpRegistry(path)
    reg.start()
    try:
        assert reg.server_names == ["tiny"]
        assert sorted(reg.tools) == ["tiny_add", "tiny_echo"]
    finally:
        reg.stop()


def test_env_interpolation_keeps_tokens_out_of_the_file(tmp_path, monkeypatch):
    """A config can be committed with `${GITHUB_TOKEN}` in it; the value comes
    from the environment at load time and never appears in the file — nor in
    the model's context, since the harness attaches it when calling."""
    monkeypatch.setenv("TEST_MCP_TOKEN", "s3cret-value")
    path = write_config(tmp_path, {"tiny": tiny(env={"UPSTREAM_TOKEN": "${TEST_MCP_TOKEN}"})})
    assert "s3cret-value" not in path.read_text(encoding="utf-8")

    reg = McpRegistry(path)
    reg.start()
    try:
        assert reg.errors == []
        assert sorted(reg.tools) == ["tiny_add", "tiny_echo"]
        # The token is not exposed through the API surface either.
        assert "s3cret-value" not in json.dumps(reg.describe())
    finally:
        reg.stop()


def test_no_config_means_no_mcp_tools():
    reg = McpRegistry(None)
    reg.start()
    assert reg.tools == {} and reg.errors == []


def test_a_missing_config_is_reported_not_fatal(tmp_path):
    reg = McpRegistry(tmp_path / "nope.json")
    reg.start()
    assert reg.tools == {}
    assert any("not found" in e for e in reg.errors)


def test_malformed_config_is_reported_not_fatal(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("{not json", encoding="utf-8")
    reg = McpRegistry(path)
    reg.start()
    assert reg.tools == {}
    assert reg.errors


def test_a_broken_server_does_not_take_the_others_down(tmp_path):
    """One misconfigured integration must not stop every other tool, or the
    API, from working."""
    path = write_config(
        tmp_path,
        {
            "broken": {"command": "/nonexistent/binary", "args": []},
            "tiny": tiny(),
        },
    )
    reg = McpRegistry(path)
    reg.start()
    try:
        assert sorted(reg.tools) == ["tiny_add", "tiny_echo"]
        assert any("failed to start" in e for e in reg.errors)
    finally:
        reg.stop()


# --- authorization --------------------------------------------------------


def test_mcp_tools_are_denied_by_default(registry):
    plain = Principal(id="plain", key_sha256=hash_key("k"))  # no allowed_tools
    catalog = ToolCatalog(registry)

    with pytest.raises(ToolNotPermittedError, match="granted explicitly"):
        catalog.resolve(["tiny_echo"], plain)

    # ...while the safe builtin still works.
    assert catalog.resolve(["get_current_time"], plain)


def test_an_exact_grant_permits_one_tool_only(registry):
    principal = Principal(
        id="narrow", key_sha256=hash_key("k"), allowed_tools=frozenset({"tiny_echo"})
    )
    catalog = ToolCatalog(registry)

    assert [t.tool_name for t in catalog.resolve(["tiny_echo"], principal)] == ["tiny_echo"]
    with pytest.raises(ToolNotPermittedError):
        catalog.resolve(["tiny_add"], principal)


def test_a_glob_grants_a_whole_server(registry):
    """The practical unit: adding a server should not mean re-enumerating
    every principal's tool list."""
    principal = Principal(id="ops", key_sha256=hash_key("k"), allowed_tools=frozenset({"tiny_*"}))
    catalog = ToolCatalog(registry)
    names = [t.tool_name for t in catalog.resolve(["tiny_echo", "tiny_add"], principal)]
    assert names == ["tiny_add", "tiny_echo"]


def test_a_glob_does_not_leak_across_servers(registry):
    principal = Principal(id="ops", key_sha256=hash_key("k"), allowed_tools=frozenset({"github_*"}))
    with pytest.raises(ToolNotPermittedError):
        ToolCatalog(registry).resolve(["tiny_echo"], principal)


def test_the_catalog_merges_builtin_and_mcp(registry):
    described = {t["name"]: t["source"] for t in ToolCatalog(registry).describe()}
    assert described["get_current_time"] == "builtin"
    assert described["http_request"] == "builtin"
    assert described["tiny_echo"] == "mcp:tiny"


def test_a_name_collision_is_reported_and_skipped(tmp_path):
    """Two servers with the same prefix would otherwise silently shadow each
    other, so the second is skipped and the operator is told to set a prefix."""
    path = write_config(tmp_path, {"a": tiny(prefix="dup"), "b": tiny(prefix="dup")})
    reg = McpRegistry(path)
    reg.start()
    try:
        assert sorted(reg.tools) == ["dup_add", "dup_echo"]
        assert any("collides" in e for e in reg.errors)
    finally:
        reg.stop()
