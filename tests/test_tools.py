"""The guarded tool implementations we kept, and why.

Strands supplies the loop, the schemas and an MCP client. What it does not
supply is address validation: its vended ``http_request`` is 35 lines and
accepts any URL with any method. These tests exist because that is the
difference between our HTTP tool and theirs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_harness.harness.tools import DANGEROUS, ToolCatalog
from model_harness.tools.guarded import (
    ToolError,
    current_time,
    http_fetch,
    truncate,
)
from model_harness.tools.net import AddressRejected, resolve_and_check

# --- SSRF guards: the reason we did not adopt the vended tool -------------


@pytest.mark.parametrize(
    "url,fragment",
    [
        ("http://127.0.0.1:8080/v1/sessions", "loopback"),
        ("http://localhost/", "loopback"),
        ("http://169.254.169.254/latest/meta-data/", "link-local"),
        ("http://10.0.0.1/admin", "private"),
        ("http://192.168.1.1/", "private"),
        ("http://172.16.0.5/", "private"),
        ("http://[::1]/", "loopback"),
        ("http://[::ffff:127.0.0.1]/", "loopback"),
        ("http://0.0.0.0/", "unspecified"),
        ("http://224.0.0.1/", "multicast"),
    ],
)
async def test_internal_addresses_are_refused(url, fragment):
    """This service lives on loopback and cloud credentials live on
    169.254.169.254 — a model-chosen URL must reach neither."""
    with pytest.raises(AddressRejected, match=fragment):
        await resolve_and_check(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "ftp://x/", "/relative"])
async def test_non_http_schemes_are_refused(url):
    with pytest.raises(AddressRejected):
        await resolve_and_check(url)


async def test_an_allowlist_refuses_everything_else():
    with pytest.raises(AddressRejected, match="allowlist"):
        await resolve_and_check("https://evil.test/", allowed_hosts=frozenset({"example.com"}))


async def test_an_unresolvable_host_is_a_readable_error():
    with pytest.raises(AddressRejected, match="resolve"):
        await resolve_and_check("https://this-host-does-not-exist.invalid/")


async def test_the_http_tool_raises_a_tool_error_for_a_blocked_address():
    with pytest.raises(ToolError, match="link-local"):
        await http_fetch("http://169.254.169.254/")


async def test_the_http_tool_requires_a_url():
    with pytest.raises(ToolError, match="url"):
        await http_fetch("")


async def test_write_methods_are_off_unless_an_operator_enables_them(monkeypatch):
    monkeypatch.delenv("HARNESS_TOOL_HTTP_ALLOW_WRITES", raising=False)
    with pytest.raises(ToolError, match="not enabled"):
        await http_fetch("https://example.com", method="POST")


# --- clock ----------------------------------------------------------------


async def test_the_clock_answers_what_a_model_cannot():
    payload = json.loads(await current_time("Europe/Amsterdam"))
    assert payload["timezone"] == "Europe/Amsterdam"
    assert "iso8601" in payload and "unix" in payload


async def test_the_clock_defaults_to_utc():
    payload = json.loads(await current_time())
    assert payload["timezone"] == "UTC"
    assert payload["utc_offset"] == "+0000"


async def test_an_unknown_timezone_is_a_readable_error():
    with pytest.raises(ToolError, match="IANA"):
        await current_time("Mars/Olympus")


def test_truncation_tells_the_model_it_happened():
    out = truncate("x" * 100, 20)
    assert out.startswith("x" * 20)
    assert "truncated" in out and "100 characters" in out
    assert truncate("short", 20) == "short"


# --- the Strands-facing layer --------------------------------------------


def test_schemas_are_generated_not_hand_written():
    """Strands derives them from type hints and the docstring."""
    catalog = {t["name"]: t for t in ToolCatalog().describe()}
    assert set(catalog) == {"get_current_time", "http_request"}
    for spec in catalog.values():
        assert spec["description"]
        assert spec["input_schema"]


def test_only_outward_reaching_tools_are_marked_dangerous():
    assert DANGEROUS == {"http_request"}
    catalog = {t["name"]: t["dangerous"] for t in ToolCatalog().describe()}
    assert catalog == {"get_current_time": False, "http_request": True}


def test_code_execution_is_not_reimplemented():
    """Strands ships real sandboxes (docker, ssh, posix), so our bare
    subprocess was a regression to keep."""
    assert "run_python" not in {t["name"] for t in ToolCatalog().describe()}
    assert not Path("src/model_harness/tools/builtin.py").exists()


def test_no_tools_requested_resolves_to_none(alice):
    catalog = ToolCatalog()
    assert catalog.resolve(None, alice) == []
    assert catalog.resolve([], alice) == []
