"""Tool registry, the built-in tools, and the SSRF guards."""

from __future__ import annotations

import json

import pytest

from model_harness.tools.base import Tool, ToolError, ToolRegistry, truncate
from model_harness.tools.builtin import (
    GET_CURRENT_TIME,
    HTTP_REQUEST,
    RUN_PYTHON,
    default_registry,
    execute,
)
from model_harness.tools.net import AddressRejected, resolve_and_check


async def _echo(args):
    return json.dumps(args)


def tool(name="echo", handler=None, **kw) -> Tool:
    return Tool(
        name=name,
        description="echo the arguments",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        handler=handler or _echo,
        **kw,
    )


# --- registry -------------------------------------------------------------


def test_registry_lookup_and_ordering():
    registry = ToolRegistry([tool("b"), tool("a")])
    assert registry.names() == ["a", "b"]  # sorted, so tool order is stable
    assert registry.get("a").name == "a"
    assert registry.get("nope") is None
    assert len(registry) == 2


def test_duplicate_registration_is_rejected():
    registry = ToolRegistry([tool("a")])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool("a"))


def test_the_default_registry_holds_the_builtins():
    assert default_registry().names() == ["get_current_time", "http_request", "run_python"]


def test_only_the_clock_is_safe_by_default():
    """Anything that can reach outside the process must be marked dangerous."""
    registry = default_registry()
    assert registry.get("get_current_time").dangerous is False
    assert registry.get("http_request").dangerous is True
    assert registry.get("run_python").dangerous is True


def test_truncation_tells_the_model_it_happened():
    out = truncate("x" * 100, 20)
    assert out.startswith("x" * 20)
    assert "truncated" in out
    assert "100 characters" in out
    assert truncate("short", 20) == "short"


# --- execute contract ----------------------------------------------------


async def test_execute_returns_output_and_timing():
    out, is_error, duration = await execute(tool(), {"a": 1})
    assert json.loads(out) == {"a": 1}
    assert is_error is False
    assert duration >= 0


async def test_a_tool_error_becomes_an_error_result_not_an_exception():
    async def failing(_args):
        raise ToolError("the 'url' argument is required")

    out, is_error, _ = await execute(tool(handler=failing), {})
    assert is_error is True
    assert "url" in out


async def test_an_unexpected_exception_is_contained():
    """A tool bug must not abort the turn and discard the loop's work."""

    async def buggy(_args):
        raise KeyError("oops")

    out, is_error, _ = await execute(tool(handler=buggy), {})
    assert is_error is True
    assert "KeyError" in out


async def test_output_is_capped_to_the_tools_limit():
    async def chatty(_args):
        return "y" * 5000

    out, is_error, _ = await execute(tool(handler=chatty, max_output_chars=100), {})
    assert is_error is False
    assert "truncated" in out
    assert len(out) < 500


# --- get_current_time ----------------------------------------------------


async def test_the_clock_tool_answers_the_question_a_model_cannot():
    out, is_error, _ = await execute(GET_CURRENT_TIME, {"timezone": "Europe/Amsterdam"})
    assert is_error is False
    payload = json.loads(out)
    assert payload["timezone"] == "Europe/Amsterdam"
    assert "iso8601" in payload and "unix" in payload


async def test_the_clock_tool_defaults_to_utc():
    payload = json.loads((await execute(GET_CURRENT_TIME, {}))[0])
    assert payload["timezone"] == "UTC"
    assert payload["utc_offset"] == "+0000"


async def test_an_unknown_timezone_is_a_readable_error():
    out, is_error, _ = await execute(GET_CURRENT_TIME, {"timezone": "Mars/Olympus"})
    assert is_error is True
    assert "IANA" in out  # tells the model what a valid value looks like


# --- SSRF guards ---------------------------------------------------------


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
        ("http://0.0.0.0/", "reserved"),
        ("http://224.0.0.1/", "multicast"),
    ],
)
async def test_internal_addresses_are_refused(url, fragment):
    """The harness itself lives on loopback, and cloud credentials live on
    169.254.169.254 — a model-chosen URL must not reach either."""
    with pytest.raises(AddressRejected, match=fragment):
        await resolve_and_check(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "ftp://x/", "/relative"])
async def test_non_http_schemes_are_refused(url):
    with pytest.raises(AddressRejected):
        await resolve_and_check(url)


async def test_an_allowlist_refuses_everything_else():
    allowed = frozenset({"example.com"})
    with pytest.raises(AddressRejected, match="allowlist"):
        await resolve_and_check("https://evil.test/", allowed_hosts=allowed)


async def test_an_unresolvable_host_is_a_readable_error():
    with pytest.raises(AddressRejected, match="resolve"):
        await resolve_and_check("https://this-host-does-not-exist.invalid/")


async def test_the_http_tool_surfaces_a_rejection_as_a_tool_error():
    """The guard must reach the model as a result, not as an exception."""
    out, is_error, _ = await execute(HTTP_REQUEST, {"url": "http://169.254.169.254/"})
    assert is_error is True
    assert "link-local" in out


async def test_the_http_tool_requires_a_url():
    out, is_error, _ = await execute(HTTP_REQUEST, {})
    assert is_error is True
    assert "url" in out


async def test_write_methods_are_off_unless_an_operator_enables_them(monkeypatch):
    monkeypatch.delenv("HARNESS_TOOL_HTTP_ALLOW_WRITES", raising=False)
    out, is_error, _ = await execute(HTTP_REQUEST, {"url": "https://example.com", "method": "POST"})
    assert is_error is True
    assert "not enabled" in out


# --- run_python ----------------------------------------------------------


async def test_python_runs_and_returns_stdout():
    out, is_error, _ = await execute(RUN_PYTHON, {"code": "print(sum(range(101)))"})
    assert is_error is False
    payload = json.loads(out)
    assert payload["stdout"].strip() == "5050"
    assert payload["exit_code"] == 0


async def test_python_reports_a_traceback_rather_than_failing_the_turn():
    out, is_error, _ = await execute(RUN_PYTHON, {"code": "raise ValueError('nope')"})
    assert is_error is False  # the tool worked; the *code* failed
    payload = json.loads(out)
    assert payload["exit_code"] != 0
    assert "ValueError" in payload["stderr"]


async def test_executed_code_cannot_see_provider_credentials(monkeypatch):
    """The child gets a bare environment, so a prompt-injected program cannot
    read the keys this service holds."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-visible")
    out, _, _ = await execute(
        RUN_PYTHON,
        {"code": "import os; print(repr(os.environ.get('ANTHROPIC_API_KEY')))"},
    )
    assert json.loads(out)["stdout"].strip() == "None"


async def test_python_requires_code():
    out, is_error, _ = await execute(RUN_PYTHON, {})
    assert is_error is True
    assert "code" in out
