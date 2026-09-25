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

from typing import Any

from strands import tool

from ..auth.principals import Principal
from ..errors import InvalidRequestError, ToolNotPermittedError
from ..tools.guarded import ToolError, current_time, http_fetch, truncate
from .mcp import McpRegistry

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
        return await current_time(timezone)
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
        return truncate(await http_fetch(url, method, body))
    except ToolError as exc:
        # Returned, not raised: the model reads the refusal and can pick a
        # different URL instead of losing the turn.
        return f"Error: {exc}"


BUILTIN = {
    "get_current_time": get_current_time,
    "http_request": http_request,
}


class ToolCatalog:
    """Everything callable this turn: our guarded tools plus every MCP tool.

    Authorization lives here rather than in Strands, which has no concept of a
    principal — it executes whatever tools the agent was handed.
    """

    def __init__(self, mcp: McpRegistry | None = None) -> None:
        self._mcp = mcp

    def _all(self) -> dict[str, Any]:
        tools = dict(BUILTIN)
        if self._mcp is not None:
            tools.update(self._mcp.tools)
        return tools

    def mcp_status(self) -> list[dict[str, Any]]:
        return self._mcp.status() if self._mcp is not None else []

    def mcp_counts(self) -> dict[str, Any]:
        if self._mcp is None:
            return {
                "configured": False,
                "servers_configured": 0,
                "servers_connected": 0,
                "servers_failed": 0,
                "servers_disabled": 0,
                "tools": 0,
            }
        return self._mcp.counts()

    def is_dangerous(self, name: str) -> bool:
        if name in DANGEROUS:
            return True
        return self._mcp is not None and name in self._mcp.tools

    def describe(self) -> list[dict[str, Any]]:
        """Catalog for ``GET /v1/tools``, built from the tools themselves."""
        out: list[dict[str, Any]] = []
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
        if self._mcp is not None:
            out.extend(self._mcp.describe())
        return out

    def resolve(self, names: list[str] | None, principal: Principal) -> list[Any]:
        """Requested names to Strands tools, enforcing our policy.

        Two gates, both required: the tool must exist, and the principal must
        be allowed it. A dangerous one — which every MCP tool is — has to be
        granted explicitly.
        """
        if not names:
            return []

        available = self._all()
        resolved: list[Any] = []

        for name in names:
            tool = available.get(name)
            if tool is None:
                raise InvalidRequestError(
                    f"Unknown tool '{name}'. See GET /v1/tools for what is available."
                )
            dangerous = self.is_dangerous(name)
            if not principal.may_use_tool(name, dangerous=dangerous):
                extra = (
                    " (it can affect state outside this service, so it must be "
                    "granted explicitly — a glob such as 'github_*' grants a "
                    "whole MCP server)"
                    if dangerous
                    else ""
                )
                raise ToolNotPermittedError(
                    f"Principal '{principal.id}' is not permitted to use tool '{name}'{extra}"
                )
            resolved.append(tool)

        # Stable order: the tool list renders before the system prompt and the
        # messages, so reordering it invalidates the provider's cached prefix.
        resolved.sort(key=lambda t: t.tool_name)
        return resolved
