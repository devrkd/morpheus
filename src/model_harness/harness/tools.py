"""Layer 2 tools: Strands-native, plus the guards worth keeping.

Strands supplies the tool loop, the schema generation (from type hints and the
docstring) and an MCP client, so our registry, our JSON schemas and our
``execute`` wrapper all go away.

Two things do *not* go away:

* **Our SSRF guards.** Strands' vended ``http_request`` is 35 lines and accepts
  any URL with any method — no address validation at all. Handing a model that
  tool means handing it cloud instance metadata at 169.254.169.254, private
  subnets, and this service on loopback. Our guarded version is kept and
  re-exposed as a Strands tool.
* **Refusing to raise at the model.** A ``ToolError`` becomes a returned string
  so the model can read it and recover, rather than aborting the turn.

Code execution is *not* re-implemented. Strands ships real sandboxes
(``strands.sandbox``: docker, ssh, posix_shell — and a module honestly named
``not_a_sandbox_local_environment``), which are strictly better than the bare
subprocess we had. Enabling one is a deployment decision, made in
``build_tools`` below rather than by writing another subprocess.
"""

from __future__ import annotations

import json
from typing import Any

from strands import tool
from strands.tools.mcp import MCPClient

from ..auth.principals import Principal
from ..errors import InvalidRequestError, ToolNotPermittedError
from ..tools.base import ToolError
from ..tools.builtin import _get_current_time, _http_request

# Tools that can reach outside this process, and therefore need an explicit
# grant on the principal rather than the default safe set.
DANGEROUS = frozenset({"http_request"})


@tool(name="get_current_time")
async def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Use this whenever the answer depends on what time it is now — the current
    date, the day of the week, how long until something, or any "today" or
    "now" question.

    Args:
        timezone: IANA timezone such as 'Europe/Amsterdam'. Defaults to UTC.
    """
    try:
        return await _get_current_time({"timezone": timezone})
    except ToolError as exc:
        return f"Error: {exc}"


@tool(name="http_request")
async def http_request(url: str, method: str = "GET", body: str | None = None) -> str:
    """Fetch a public URL over HTTP or HTTPS and return the status and body.

    Use this to read documentation, call a public API, or check a web page.
    Only public internet addresses work: requests to localhost, private
    networks, and cloud metadata endpoints are refused. Large bodies are
    truncated, so prefer specific URLs over whole sites.

    Args:
        url: Absolute http(s) URL to fetch.
        method: HTTP method. Only GET and HEAD unless writes are enabled.
        body: Request body, for write methods when enabled.
    """
    try:
        return await _http_request({"url": url, "method": method, "body": body})
    except ToolError as exc:
        # Returned, not raised: the model reads the refusal and can pick a
        # different URL instead of losing the turn.
        return f"Error: {exc}"


BUILTIN = {
    "get_current_time": get_current_time,
    "http_request": http_request,
}


def describe() -> list[dict[str, Any]]:
    """Catalog for ``GET /v1/tools``, built from the decorated tools."""
    out = []
    for name, fn in sorted(BUILTIN.items()):
        spec = fn.tool_spec
        out.append(
            {
                "name": name,
                "description": (spec.get("description") or "").strip(),
                "input_schema": spec.get("inputSchema", {}),
                "dangerous": name in DANGEROUS,
                "source": "builtin",
            }
        )
    return out


def resolve(names: list[str] | None, principal: Principal) -> list[Any]:
    """Turn requested tool names into Strands tools, enforcing our policy.

    Strands does not know about principals, so authorization stays here — the
    same two gates as before: the tool must exist, and a dangerous one must be
    granted explicitly.
    """
    if not names:
        return []

    resolved = []
    for name in names:
        fn = BUILTIN.get(name)
        if fn is None:
            raise InvalidRequestError(
                f"Unknown tool '{name}'. See GET /v1/tools for what is available."
            )
        dangerous = name in DANGEROUS
        if not principal.may_use_tool(name, dangerous=dangerous):
            extra = (
                " (it can affect state outside this service, so it must be granted explicitly)"
                if dangerous
                else ""
            )
            raise ToolNotPermittedError(
                f"Principal '{principal.id}' is not permitted to use tool '{name}'{extra}"
            )
        resolved.append(fn)

    # Stable order: the tool list renders before the system prompt and the
    # messages, so reordering it invalidates the provider's cached prefix.
    resolved.sort(key=lambda f: f.tool_name)
    return resolved


def mcp_clients(config_json: str | None) -> list[MCPClient]:
    """Build MCP clients from configuration.

    Each entry is ``{"url": ..., "headers": {...}, "prefix": ...}``. Headers are
    where credential brokering happens: the token lives in this service's
    configuration and is attached at call time, so the model never holds a
    secret it could leak into a transcript.
    """
    if not config_json:
        return []
    try:
        entries = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"HARNESS_MCP_SERVERS is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        # ValueError, not TypeError: every failure here is a malformed-config
        # error that startup reports as one class.
        raise ValueError(  # noqa: TRY004
            "HARNESS_MCP_SERVERS must be a JSON list of server objects"
        )

    clients = []
    for entry in entries:
        if not isinstance(entry, dict) or "url" not in entry:
            raise ValueError("Each MCP server entry needs at least a 'url'")
        clients.append(
            MCPClient(
                url=entry["url"],
                headers=entry.get("headers"),
                prefix=entry.get("prefix"),
            )
        )
    return clients
