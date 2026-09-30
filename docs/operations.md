# Operating it

The operator's guide: keys and principals, tools and permissions, MCP configuration, health
and monitoring, the configuration surface, and deployment notes.

## Keys and principals

A **principal** is whoever or whatever calls the service — a team, a service account, a person.
Keys are minted by an operator:

```bash
morpheus mint-key --id team-exchange \
  --models claude-sonnet-5,gpt-5 --max-tokens-per-turn 4000
```

This prints the key **once** (it is not recoverable — losing it means minting a new one) plus
the JSON entry to paste into the file named by `HARNESS_PRINCIPALS_FILE`:

```json
{
  "principals": [
    {
      "id": "team-exchange",
      "key_sha256": "cefb910c…",
      "allowed_models": ["claude-sonnet-5", "gpt-5"],
      "max_tokens_per_turn": 4000
    },
    { "id": "platform-team", "key_sha256": "1111…" },
    { "id": "retired-service", "key_sha256": "2222…", "disabled": true }
  ]
}
```

Only the SHA-256 digest is stored. Omit `allowed_models` to permit the whole catalog.

**What a principal buys you:**

| Control | Effect |
|---|---|
| **Model allowlist** | Enforced against the *canonical* model id — an alias cannot slip past a list that omits its model. `GET /v1/models` is filtered to match. |
| **Session ownership** | Every session records its creator; foreign access gets `404`, not `403`. |
| **Per-turn token ceiling** | Caps this principal after the model's own ceiling; binding is reported in `adjustments`. |
| **Revocation** | `disabled: true`, or delete the entry, then restart. Neither touches a provider key. |

**Startup fails closed.** With neither a principals file nor `HARNESS_ALLOW_ANONYMOUS=true`,
the service refuses to start — one forgotten environment variable must never turn it into an
open proxy for credentials that cost money. Anonymous mode shares a single `anonymous`
principal across all callers (and all sessions). Localhost only.

## Tools and permissions

Two independent gates: the tool must **exist** and be **granted** to the principal.

- Tool names are `<prefix>_<tool>` — the server's configured `prefix` plus the server's own
  tool name.
- Grant by exact name or by glob, per principal:

```bash
scripts/grant-tools.sh local-dev github_get_commit    # one tool
scripts/grant-tools.sh local-dev 'github_*'           # the whole server
scripts/grant-tools.sh --revoke local-dev github_create_issue
scripts/grant-tools.sh --list                         # who has what
```

This edits the principals file (equivalent to `"allowed_tools": ["github_get_commit"]`).
**Restart the server afterwards** — principals are read at startup.

Rules worth internalising:

- **Every MCP tool is dangerous by default** — none is granted unless explicitly named, even
  in anonymous mode. A flag turns off authentication; it does not confer capability.
- A glob is the practical unit — adding a server should not mean re-enumerating every
  principal's tool list. `"*"` grants everything; treat it as an operator principal.
- Revoking an exact name does **not** override a glob that still covers it — the grant script
  warns when that happens, because believing you removed access when you did not is the
  expensive mistake.
- Read names from `GET /v1/tools` rather than guessing; a name that does not exist is a
  `400 Unknown tool`. A server's `tool_filters` decides which tools *exist*; `allowed_tools`
  decides who may call them. Filtering a tool out means no grant can reach it.

### The guarded HTTP tool

`http_request` refuses internal, loopback, private, and cloud-metadata addresses — validated
against the **resolved IPs** (a hostname can point anywhere) and re-validated on **every
redirect hop** (the redirect target is chosen by the remote server). Further narrowing:

- `HARNESS_TOOL_HTTP_ALLOWED_HOSTS` restricts outbound fetches to specific hosts.
- `HARNESS_TOOL_HTTP_ALLOW_WRITES=false` (default) keeps the tool read-only.

## MCP servers

Point `HARNESS_MCP_CONFIG` at a standard `mcpServers` JSON file — the same format Claude
Desktop and `.mcp.json` use, so an existing config works unchanged:

```json
{
  "mcpServers": {
    "github": {
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": { "Authorization": "Bearer ${GITHUB_TOKEN}" },
      "prefix": "github",
      "tool_filters": { "allowed": ["get_*", "list_*", "search_*"] }
    },
    "gdrive": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-gdrive"],
      "env": { "GDRIVE_CREDENTIALS_PATH": "${GDRIVE_CREDENTIALS_PATH}" },
      "prefix": "gdrive"
    }
  }
}
```

- Transport is inferred: `command` means stdio, `url` means streamable HTTP.
- `disabled: true` keeps a server configured but off.
- **Secrets never enter the config.** `${VAR}` is interpolated from the environment at load
  time, so the file is safe to commit; tokens stay in the service.
- Servers load **independently at startup**; one missing token disables only its own server and
  the error names the variable. Avoid `continue_on_error: true` — it makes the skip silent.

## Health and monitoring

- `GET /v1/health` (open): `status` (`ok` / `degraded`), per-provider availability and
  credential *source* (a name, never a value), and MCP **counts only** — server names and error
  text stay behind auth on `GET /v1/tools`, because a server name can be an internal hostname.
- `GET /v1/tools` (authenticated): per-server state dot (connected / failed / disabled),
  transport, tool counts, and the failure reason when there is one.
- Every converse response carries `usage` and `latency_ms` — the raw material for spend and
  latency accounting.

Provider availability semantics differ on purpose, following each SDK: an OpenAI adapter with
no credential reports itself unavailable at startup and requests return `503` immediately; the
Anthropic adapter resolves credentials lazily and lets the request decide. When nothing
resolves at request time, the error is translated into the same clean `503`.

## Configuration surface

Read from environment variables (and a local `.env`). The essentials:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Outbound overrides; leaving one unset does **not** disable its provider — the SDK walks its credential chain |
| `HARNESS_PRINCIPALS_FILE` | Path to the principals JSON (required unless anonymous mode) |
| `HARNESS_ALLOW_ANONYMOUS` | Run with no inbound auth (localhost only; shared principal) |
| `HARNESS_MCP_CONFIG` | Path to the `mcpServers` JSON |
| `HARNESS_HOST` / `HARNESS_PORT` | Bind address (default `127.0.0.1` / `8080`) |
| `HARNESS_SESSION_DIR` | Where transcripts and the ownership index live |
| `HARNESS_SESSION_TTL_SECONDS` | Session expiry (default 86400; 0 disables) |
| `HARNESS_SESSION_MAX_TURNS` | Hard cap on stored turns per session (default 200; 0 disables) |
| `HARNESS_REQUEST_TIMEOUT_SECONDS` | Per-call timeout handed to the provider SDKs |
| `HARNESS_TURN_TIMEOUT_SECONDS` | Wall-clock ceiling on a whole turn, tool loop included |
| `HARNESS_TOOL_HTTP_ALLOWED_HOSTS` / `HARNESS_TOOL_HTTP_ALLOW_WRITES` | Narrow the `http_request` tool |

## Deployment notes

- **Terminate TLS in front of the service.** Inbound keys are bearer tokens, only as safe as
  the channel. Keep `HARNESS_HOST` on loopback otherwise.
- **Run one worker.** Sessions live in the process; a second worker would not see sessions the
  first created. Horizontal scaling means moving transcripts and the ownership index to a
  shared backend.
- **Prefer workload identity over long-lived keys** for outbound credentials on a real server:
  short-lived, auto-refreshed, nothing static on disk.
- **Principals and MCP servers load at startup.** Key grants/revocations and MCP config changes
  require a restart — minting a credential is an operator action by design.
- **A failing MCP server must not take the API down.** Servers are loaded independently; a
  server that fails to start is logged and skipped.

See [Architecture → Known limitations](architecture.md#known-limitations) for the full list of
design boundaries before planning a production rollout.
