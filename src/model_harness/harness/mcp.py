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
from pathlib import Path
from typing import Any

from strands.tools.mcp import MCPClient

logger = logging.getLogger("model_harness.mcp")


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
        self.errors: list[str] = []

    @property
    def tools(self) -> dict[str, Any]:
        return self._tools

    def server_for(self, tool_name: str) -> str | None:
        return self._servers.get(tool_name)

    @property
    def server_names(self) -> list[str]:
        return sorted(set(self._servers.values()))

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Connect to every enabled server and discover its tools.

        A server that fails to start is recorded and skipped rather than
        taking the service down with it. One misconfigured integration should
        not stop every other tool, or the API, from working.
        """
        if self._config_path is None:
            return
        if not self._config_path.exists():
            self.errors.append(f"MCP config not found: {self._config_path}")
            logger.warning("%s", self.errors[-1])
            return

        try:
            clients = MCPClient.load_servers(str(self._config_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.errors.append(f"MCP config {self._config_path} is unusable: {exc}")
            logger.warning("%s", self.errors[-1])
            return

        names = self._configured_names()

        for index, client in enumerate(clients):
            label = names[index] if index < len(names) else f"server_{index}"
            started = False
            try:
                client.start()
                started = True
                discovered = client.list_tools_sync()
            except Exception as exc:  # noqa: BLE001 - any transport failure
                self.errors.append(f"MCP server '{label}' failed to start: {exc}")
                logger.warning("%s", self.errors[-1])
                # Only stop what actually started; stopping a client that never
                # connected leaves its shutdown coroutine unawaited.
                if started:
                    self._safe_stop(client)
                continue

            self._clients.append(client)
            for tool in discovered:
                if tool.tool_name in self._tools:
                    self.errors.append(
                        f"MCP tool '{tool.tool_name}' from '{label}' collides with an "
                        "existing tool and was skipped; set a distinct 'prefix'"
                    )
                    logger.warning("%s", self.errors[-1])
                    continue
                self._tools[tool.tool_name] = tool
                self._servers[tool.tool_name] = label

            logger.info(
                "MCP server '%s': %d tool(s) — %s",
                label,
                len(discovered),
                ", ".join(t.tool_name for t in discovered) or "none",
            )

    def stop(self) -> None:
        for client in self._clients:
            self._safe_stop(client)
        self._clients.clear()
        self._tools.clear()
        self._servers.clear()

    @staticmethod
    def _safe_stop(client: MCPClient) -> None:
        try:
            # stop() carries the __exit__ signature, since MCPClient is also a
            # context manager.
            client.stop(None, None, None)
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.debug("MCP client stop failed: %s", exc)

    # --- helpers ----------------------------------------------------------

    def _configured_names(self) -> list[str]:
        """Server names in config order.

        ``load_servers`` returns clients but not their names, and the order
        matches the enabled servers in the file, so the names are recovered
        here to label tools and error messages.
        """
        try:
            raw = json.loads(self._config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        servers = raw.get("mcpServers", raw)
        if not isinstance(servers, dict):
            return []
        return [
            name
            for name, cfg in servers.items()
            if not (isinstance(cfg, dict) and cfg.get("disabled"))
        ]

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
