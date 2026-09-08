"""MCP servers as a configurable tool source.

Tools stop being something we write and become something an operator
configures. A GitHub server, a Google Drive server, an internal one — each
contributes its tools to the catalog, and the harness's job narrows to
authorization, naming and lifecycle.

Three decisions worth stating:

**The config format is not ours.** ``MCPClient.load_servers`` reads the
standard ``mcpServers`` shape — the same file Claude Desktop and ``.mcp.json``
use — including stdio (``command``/``args``/``env``) and HTTP (``url``/
``headers``), ``disabled``, and ``${VAR}`` interpolation. Inventing a format
here would mean every operator hand-translating configs that already exist.

**Secrets stay in the environment.** Because interpolation happens at load
time, a config file can be committed with ``"Authorization": "Bearer ${GITHUB_TOKEN}"``
and the token never appears in it. It also never reaches the model: the
harness holds it and attaches it when calling the server, so a prompt
injection cannot exfiltrate what the model never had.

**Every MCP tool is dangerous.** It runs code we did not write, against a
system we do not control, with credentials this service holds. So none is
granted by default — a principal names them, or a whole server with a
``github_*`` wildcard.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strands.tools.mcp import MCPClient

logger = logging.getLogger("model_harness.mcp")


@dataclass
class ServerStatus:
    """What became of one configured server.

    Kept for every server in the file, not just the ones that worked — a
    server that is off or broken is exactly what someone checking status
    needs to see. "Configured but absent from the list" is the one answer
    that helps nobody.
    """

    name: str
    state: str
    """One of: connected, failed, disabled."""

    transport: str = "unknown"
    tools: list[str] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "transport": self.transport,
            "tool_count": len(self.tools),
            "tools": sorted(self.tools),
            # Config values (a command path, a URL), never an interpolated
            # secret — those live in headers and env, which are not echoed.
            "error": self.error,
        }


class McpRegistry:
    """Owns the MCP client connections and the tools they expose.

    Connections are opened once at startup, not per request: an MCP server is
    a session, and paying stdio process spawn or an HTTP handshake on every
    turn would dominate the latency of a short one.
    """

    def __init__(self, config_path: Path | None) -> None:
        self._config_path = config_path
        self._clients: list[MCPClient] = []
        self._tools: dict[str, Any] = {}
        self._servers: dict[str, str] = {}
        self._status: dict[str, ServerStatus] = {}
        self.errors: list[str] = []

    @property
    def tools(self) -> dict[str, Any]:
        return self._tools

    def server_for(self, tool_name: str) -> str | None:
        return self._servers.get(tool_name)

    @property
    def server_names(self) -> list[str]:
        return sorted(set(self._servers.values()))

    @property
    def configured(self) -> bool:
        return self._config_path is not None

    def status(self) -> list[dict[str, Any]]:
        """Per-server status, for an authenticated caller."""
        return [self._status[name].summary() for name in sorted(self._status)]

    def counts(self) -> dict[str, Any]:
        """Aggregate counts only — safe for the unauthenticated health route.

        Server names and error text stay out: a name can be an internal
        hostname, and health is open by design.
        """
        states = [st.state for st in self._status.values()]
        return {
            "configured": self._config_path is not None,
            "servers_configured": len(states),
            "servers_connected": states.count("connected"),
            "servers_failed": states.count("failed"),
            "servers_disabled": states.count("disabled"),
            "tools": len(self._tools),
        }

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Connect to every enabled server and discover its tools.

        Servers are loaded **one at a time**, deliberately. Handing
        ``load_servers`` the whole file returns a flat client list with no
        names attached, so a server it silently skips (a missing env var, say)
        shifts every later index and tools get attributed to the wrong server.
        One call per server keeps names exact and makes each failure
        attributable to the thing that failed.
        """
        if self._config_path is None:
            return
        if not self._config_path.exists():
            self.errors.append(f"MCP config not found: {self._config_path}")
            logger.warning("%s", self.errors[-1])
            return

        try:
            servers = self._read_config()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.errors.append(f"MCP config {self._config_path} is unusable: {exc}")
            logger.warning("%s", self.errors[-1])
            return

        for name, cfg in servers.items():
            transport = cfg.get("transport", "stdio" if cfg.get("command") else "http")

            if cfg.get("disabled"):
                self._status[name] = ServerStatus(name=name, state="disabled", transport=transport)
                continue

            self._start_one(name, cfg, transport)

    def _start_one(self, name: str, cfg: dict[str, Any], transport: str) -> None:
        """Bring up one server, recording whatever happens to it."""

        def fail(reason: str) -> None:
            self.errors.append(f"MCP server '{name}': {reason}")
            logger.warning("%s", self.errors[-1])
            self._status[name] = ServerStatus(
                name=name, state="failed", transport=transport, error=reason
            )

        try:
            clients = MCPClient.load_servers({name: cfg})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # Most often a missing ${VAR}. Reported against this server only,
            # so one unset token does not disable every other integration.
            fail(str(exc))
            return

        if not clients:
            fail("its configuration produced no client")
            return

        client = clients[0]
        started = False
        try:
            client.start()
            started = True
            discovered = client.list_tools_sync()
        except Exception as exc:  # noqa: BLE001 - any transport failure
            fail(f"failed to start: {exc}")
            # Only stop what actually started; stopping a client that never
            # connected leaves its shutdown coroutine unawaited.
            if started:
                self._safe_stop(client)
            return

        self._clients.append(client)
        mine: list[str] = []
        for tool in discovered:
            if tool.tool_name in self._tools:
                self.errors.append(
                    f"MCP tool '{tool.tool_name}' from '{name}' collides with an "
                    "existing tool and was skipped; set a distinct 'prefix'"
                )
                logger.warning("%s", self.errors[-1])
                continue
            self._tools[tool.tool_name] = tool
            self._servers[tool.tool_name] = name
            mine.append(tool.tool_name)

        self._status[name] = ServerStatus(
            name=name, state="connected", transport=transport, tools=mine
        )
        logger.info(
            "MCP server '%s' (%s): %d tool(s) — %s",
            name,
            transport,
            len(mine),
            ", ".join(mine) or "none",
        )

    def stop(self) -> None:
        for client in self._clients:
            self._safe_stop(client)
        self._clients.clear()
        self._tools.clear()
        self._servers.clear()
        self._status.clear()

    @staticmethod
    def _safe_stop(client: MCPClient) -> None:
        try:
            # stop() carries the __exit__ signature, since MCPClient is also a
            # context manager.
            client.stop(None, None, None)
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.debug("MCP client stop failed: %s", exc)

    # --- helpers ----------------------------------------------------------

    def _read_config(self) -> dict[str, dict[str, Any]]:
        """The server table, cleaned up for consumption.

        Keys beginning with ``_`` are treated as comments — that is how the
        example config annotates itself, and a bare string there would
        otherwise be rejected as a malformed server.

        ``continue_on_error`` is deliberately *not* forced on. It makes
        Strands skip a broken server and return nothing, which swallows the
        reason: "its configuration produced no client" instead of
        "environment variable 'GITHUB_TOKEN' is not set". Loading one server
        at a time already gives the isolation, so letting the error surface
        keeps the message actionable.
        """
        raw = json.loads(self._config_path.read_text(encoding="utf-8"))
        table = raw.get("mcpServers", raw)
        if not isinstance(table, dict):
            # ValueError throughout: every failure here is one class of
            # malformed-config error that start() reports the same way.
            raise ValueError(  # noqa: TRY004
                "expected an object of server name -> config"
            )

        cleaned: dict[str, dict[str, Any]] = {}
        for name, cfg in table.items():
            if name.startswith("_"):
                continue
            if not isinstance(cfg, dict):
                raise ValueError(f"server '{name}' must be an object")  # noqa: TRY004
            cleaned[name] = dict(cfg)
        return cleaned

    def describe(self) -> list[dict[str, Any]]:
        """Catalog entries for ``GET /v1/tools``."""
        out = []
        for name in sorted(self._tools):
            spec = self._tools[name].tool_spec
            out.append(
                {
                    "name": name,
                    "description": (spec.get("description") or "").strip(),
                    "input_schema": spec.get("inputSchema", {}),
                    # Runs code we did not write, against a system we do not
                    # control, with credentials this service holds.
                    "dangerous": True,
                    "source": f"mcp:{self._servers.get(name, 'unknown')}",
                }
            )
        return out
