# morpheus

A Bedrock-style unified inference service. One HTTP API, one model id per
request; the harness picks the provider and translates both ways. Ask for
`claude-opus-5` and it goes to Anthropic; ask for `gpt-5` and it goes to
OpenAI. Conversation context lives on the server, so a session can move
between providers mid-conversation and the new model still sees the whole
history.

```
                      ┌──────────────────────────────────────┐
  POST /v1/converse   │  route(model) ──> provider adapter   │
  { model, messages,  │       │                              │
    session_id }      │       └──> session store (canonical  │──> Anthropic
        ───────────►  │             transcript)              │──> OpenAI
                      └──────────────────────────────────────┘
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env

# Outbound: how the harness reaches the providers.
# An API key is the simplest option; see "Authentication" for the alternatives.
export ANTHROPIC_API_KEY=sk-ant-...

# Inbound: who may call the harness. Startup fails without this.
.venv/bin/python -m morpheus mint-key --id me > /tmp/minted
# paste the printed entry into ./principals.json, then:
export HARNESS_PRINCIPALS_FILE=./principals.json

.venv/bin/python -m morpheus
```

Interactive docs at `http://127.0.0.1:8080/docs`.

Then, from a second terminal, exercise the whole thing — health, the filtered
catalog, a two-provider session, the stored transcript, and the auth checks:

```bash
scripts/demo.sh
```

Steps 1, 2 and 6 of that script work with no provider credential at all, so
it is also a useful check that the service is wired up before you spend
anything.

For a throwaway local run, `HARNESS_ALLOW_ANONYMOUS=true` skips inbound auth
entirely. Startup refuses to proceed with neither that flag nor a principals
file — see [Authentication](#authentication) for why.

## Web client

The service serves a small chat UI at **`http://127.0.0.1:8080/app/`** (and
`/` redirects there). Sign in with an inbound harness key — the same key the
API takes — then create sessions and chat in them.

It is served from the harness itself rather than as a separate page, because a
browser page on another origin could not call the API without CORS, and
opening CORS on a service holding provider credentials is a bigger decision
than a demo UI warrants. Same origin means no CORS, no preflight, no second
process. Bearer tokens help here too: unlike cookies, a browser never attaches
an `Authorization` header on its own, so there is no CSRF surface.

What it demonstrates, which is the point:

- **The transcript lives server-side.** Opening a session reloads it from
  `GET /v1/sessions/{id}`, so a refresh, a new tab, or another machine picks
  up the same conversation.
- **Only the newest turn is uploaded.** Each reply shows its input-token
  count, so you can watch the replayed context grow turn by turn — and watch
  `cached` appear once a session is long enough to hit the prompt cache.
- **Sessions are per-principal.** Two different keys are two different users
  with disjoint session lists, which is the ownership check doing its job.
- **`adjustments` are surfaced**, so a clamped effort or a dropped sampling
  parameter is visible rather than silent.

The key is kept in `sessionStorage`, so closing the tab discards it. That is
still a bearer token in a browser — fine for local development, and the reason
`HARNESS_HOST` defaults to loopback.

## The API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/converse` | required | Send a turn, get the whole response |
| `POST` | `/v1/converse-stream` | required | Same, streamed as SSE |
| `GET` | `/v1/models` | required | Capability catalog, filtered to your allowlist |
| `GET` | `/v1/whoami` | required | Your principal id and limits; the cheapest way to validate a key |
| `GET` | `/v1/sessions` | required | Your session ids, or summaries with `?detail=true` |
| `GET` | `/v1/sessions/{id}` | required | Session metadata (`?include_messages=true` for the transcript) |
| `DELETE` | `/v1/sessions/{id}` | required | Forget a session |
| `GET` | `/v1/health` | **open** | Liveness plus per-provider credential status |

`/v1/health` is the one unauthenticated route, so a load balancer or
deployment check can reach it. It returns which credential *source* each
provider resolved — a source name, never a value — and no transcript.

### One conversation, two providers

Send only the new turn. The stored transcript is prepended for you.

```bash
# Turn 1 — Anthropic. No session_id, so one is created and returned.
curl -s localhost:8080/v1/converse \
  -H "authorization: Bearer $HARNESS_KEY" \
  -H 'content-type: application/json' -d '{
  "model": "claude-opus-5",
  "system": [{"text": "You are concise."}],
  "messages": [{"role": "user", "content": [{"type": "text", "text": "My name is Ramesh."}]}],
  "effort": "low"
}'
# -> { "session_id": "sess_ab12...", "provider": "anthropic", ... }

# Turn 2 — OpenAI, same session. GPT-5 sees turn 1.
curl -s localhost:8080/v1/converse \
  -H "authorization: Bearer $HARNESS_KEY" \
  -H 'content-type: application/json' -d '{
  "model": "gpt-5",
  "session_id": "sess_ab12...",
  "messages": [{"role": "user", "content": [{"type": "text", "text": "What is my name?"}]}]
}'
```

The response is the same shape whichever provider served it:

```json
{
  "session_id": "sess_ab12...",
  "provider": "openai",
  "model": "gpt-5",
  "native_model": "gpt-5-2025-08-07",
  "output": {"message": {"role": "assistant", "content": [{"type": "text", "text": "Ramesh."}]}},
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 62, "output_tokens": 4, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "reasoning_tokens": 128},
  "latency_ms": 1841,
  "adjustments": ["provider switched anthropic -> openai: ..."]
}
```

### `adjustments`: nothing is silently altered

Every neutral parameter the target model could not take verbatim is reported
back. The harness never forwards a parameter that would earn a 400, and never
hides that it dropped one.

```json
"adjustments": [
  "dropped temperature, top_p: not accepted by claude-opus-5",
  "effort 'max' clamped to 'high' (highest supported by gpt-5)",
  "max_tokens 100000 lowered to 16384, the output ceiling for gpt-4o"
]
```

### Streaming

`POST /v1/converse-stream` returns SSE with a canonical event vocabulary —
`message_start`, `content_block_start`, `content_delta`, `content_block_stop`,
`message_stop`, `error` — then a final `data: [DONE]`. The session id is in
the `X-Session-Id` response header, available before the first body byte.

Because the HTTP status is already sent once a stream starts, a mid-stream
failure arrives as an `error` event rather than a status code. Treat it as
terminal.

## Authentication

Two independent directions. Confusing them is the usual way a gateway like
this ends up insecure.

### Inbound: callers → harness

A caller presents a harness-issued key as `Authorization: Bearer mh_…`. Keys
are minted by an operator and stored only as a SHA-256 digest:

```bash
morpheus mint-key --id team-exchange \
  --models claude-sonnet-5,gpt-5 --max-tokens-per-turn 4000
```

That prints the key **once** — it is not recoverable — plus the JSON entry to
paste into the file named by `HARNESS_PRINCIPALS_FILE`:

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

Omit `allowed_models` to permit the whole catalog. See
[`principals.example.json`](principals.example.json).

SHA-256 rather than bcrypt or argon2 is deliberate: these keys are 256 bits of
machine-generated randomness, so there is no dictionary to attack and nothing
for a slow KDF to buy. A slow KDF is for human-chosen passwords.

What a principal buys you:

| Control | Effect |
|---|---|
| **Model allowlist** | Enforced against the *canonical* model id, so an alias like `opus` cannot slip past a list that omits `claude-opus-5`. `GET /v1/models` is filtered to match. |
| **Session ownership** | Every session records its creator. Another principal reading, deleting, or appending to it gets `404`, not `403` — a `403` would confirm the id exists. |
| **Per-turn token ceiling** | `max_tokens_per_turn` caps this principal after the model's own ceiling, reported in `adjustments`. |
| **Revocation** | `disabled: true`, or delete the entry. Neither touches a provider key. |

**Startup fails when inbound auth is unconfigured.** Neither a principals file
nor `HARNESS_ALLOW_ANONYMOUS=true` is an error, not a default:

```
error: Inbound authentication is not configured. Either set
HARNESS_PRINCIPALS_FILE to a principals file (mint a key with
`morpheus mint-key --id <name>`), or set HARNESS_ALLOW_ANONYMOUS=true to
run with no authentication — which lets anyone who can reach this port spend
the provider credentials.
```

Failing closed matters here because the failure is silent otherwise: one
forgotten environment variable would turn this into an open proxy for
credentials that cost real money. Anonymous mode remains available, and every
anonymous caller shares one `anonymous` principal — so they also share session
ownership, and can read each other's transcripts. Localhost only.

### Outbound: harness → providers

Each adapter lets its SDK resolve its own credential, because both SDKs
implement a chain that is richer than one environment variable.

Anthropic, first match wins:

```
ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN → ANTHROPIC_PROFILE / active
`ant auth login` profile → workload identity federation → default profile
in ~/.config/anthropic
```

OpenAI is narrower: an API key, or workload identity via the SDK's own env
vars. On a real server prefer **workload identity** — short-lived,
auto-refreshed, no long-lived secret on disk — over a static key.

Only explicitly configured values are passed to a client constructor;
everything omitted leaves the SDK free to walk its chain. Falsy values are
dropped rather than forwarded, because an exported-but-empty
`ANTHROPIC_API_KEY` would otherwise shadow the entire chain — a real footgun,
covered by a test.

`GET /v1/health` reports which link was found:

```json
{
  "status": "ok",
  "providers": {
    "anthropic": {"available": true, "credential_source": "oauth_profile",
                  "credential_detected": true, "note": "profile 'work'"},
    "openai": {"available": false, "credential_source": "undetected",
               "credential_detected": false,
               "note": "the OpenAI SDK found no usable credential at startup"}
  }
}
```

Detection is best-effort, and the two adapters report availability
differently on purpose, following their SDKs:

- **OpenAI** raises at construction with no credential, so that answer is
  authoritative — the adapter reports itself unavailable and a request returns
  `503` immediately.
- **Anthropic** constructs regardless and resolves lazily, so the adapter
  stays available and lets the request decide. Refusing up front would reject
  work that a profile or workload identity would have authenticated. When
  nothing resolves at all, the SDK signals it with a bare `TypeError` at
  request time, which the adapter translates into the same clean `503`.

### Error detail and `error_id`

Errors about the caller's own request explain themselves. Errors originating
inside a provider do not — the SDK exception behind them carries provider
response bodies and account detail — so they return a generic message plus an
`error_id`:

```json
{"error": {"code": "provider_rejected_request",
           "message": "The upstream model provider rejected this request. Quote the error_id when reporting it.",
           "provider": "anthropic", "retryable": false,
           "error_id": "err_0b1a16ce7854"}}
```

The full detail is logged against that id, so a report quoting it leads
straight to the context. An invalid inbound key and a disabled one return
byte-identical responses, so the API cannot be used to enumerate valid keys.

## Tools

Without tools a model can only talk: no clock, no network, no way to compute.
Tools come from two places.

**Built in** — two, kept for specific reasons:

| Tool | Dangerous | Why it is ours |
|---|---|---|
| `get_current_time` | no | The most common thing a model is asked and cannot know. Strands ships no clock. |
| `http_request` | yes | Strands' vended `http_request` is 35 lines and accepts any URL with any method — no address validation at all. Ours refuses internal addresses. |

Code execution is deliberately absent: `strands.sandbox` ships docker, ssh and
posix sandboxes, so writing another bare subprocess would be a regression.

**From MCP servers** — configured, not written. See below.

Request tools per turn; the response reports what ran:

```bash
curl -s localhost:8080/v1/converse \
  -H "authorization: Bearer $(cat .harness-key)" \
  -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-5",
       "tools":["get_current_time","github_search_issues"],
       "messages":[{"role":"user","content":[{"type":"text","text":"Any open issues about auth?"}]}]}'
```

`tool_calls` lists every execution, `iterations` counts provider round trips,
and `usage` is summed across the whole loop — a tool turn costs every request
in it, not just the last.

`GET /v1/tools` lists everything with its schema, its `source`
(`builtin` or `mcp:<server>`), and whether you may use it.

### MCP servers

Point `HARNESS_MCP_CONFIG` at a **standard `mcpServers` JSON file** — the same
format Claude Desktop and `.mcp.json` use, so an existing config works
unchanged. See [`mcp.example.json`](mcp.example.json).

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

Transport is inferred: `command` means stdio, `url` means streamable HTTP.
`disabled: true` keeps a server configured but off. `tool_filters` narrows a
server's surface — the `allowed` list above makes GitHub read-only.

### Seeing whether they are up

The web client's sidebar has a **Tools** panel: one group per configured
server with a status dot — green connected, red failed, grey disabled — its
transport and tool count, the failure reason inline when there is one, and a
checkbox per tool that enables it for the next message. A tool you have not
been granted shows as `not granted` rather than being hidden, so you can see
what to ask an operator for.

The same thing over HTTP:

```bash
# authenticated: names, transports, tool lists, and why a server failed
curl -s localhost:8080/v1/tools -H "authorization: Bearer $(cat .harness-key)" | jq .mcp_servers
```
```json
[
  {"name": "github", "state": "failed", "transport": "http", "tool_count": 0,
   "tools": [], "error": "environment variable 'GITHUB_TOKEN' is not set"},
  {"name": "off", "state": "disabled", "transport": "stdio", "tool_count": 0,
   "tools": [], "error": null},
  {"name": "tiny", "state": "connected", "transport": "stdio", "tool_count": 2,
   "tools": ["tiny_add", "tiny_echo"], "error": null}
]
```

`GET /v1/health` carries **counts only** — `servers_connected`,
`servers_failed`, `servers_disabled`, `tools`. That route is unauthenticated,
and a server name can be an internal hostname, so names and error text stay
behind auth.

Servers are loaded **independently**, so one missing token disables only its
own server and the error names the variable. Avoid `continue_on_error: true`
for that reason: it makes the skip silent, replacing "environment variable
'GITHUB_TOKEN' is not set" with nothing useful.

Servers connect **once at startup**, not per request: an MCP server is a
session, and paying stdio process spawn or an HTTP handshake on every turn
would dominate a short turn's latency. A server that fails to start is logged
and skipped — one misconfigured integration must not take the API down with
it.

**Secrets never enter the config, or the model.** `${VAR}` is interpolated
from the environment at load time, so the file is safe to commit. The token
stays in this service and is attached when calling the server, so a prompt
injection cannot exfiltrate what the model never had.

### Permissions

Two gates, both required: the tool must exist, and the principal must be
allowed it.

**Every MCP tool counts as dangerous** — it runs code we did not write,
against a system we do not control, with credentials this service holds. So
none is granted by default, and that holds *even in anonymous mode*: a flag
turns off authentication, it does not confer capability. Using MCP tools
therefore requires a principals file, which is the point rather than a
limitation.

Tools are named `<prefix>_<tool>` — the `prefix` from the server's config plus
the server's own tool name — so `get_commit` from a server prefixed `github`
becomes `github_get_commit`. Grant by exact name or by glob:

```bash
scripts/grant-tools.sh local-dev github_get_commit    # one tool
scripts/grant-tools.sh local-dev 'github_*'           # the whole server
scripts/grant-tools.sh --revoke local-dev github_create_issue
scripts/grant-tools.sh --list                         # who has what
```

That edits the principals file, which is equivalent to:

```json
{ "id": "team-a", "allowed_tools": ["github_get_commit", "gdrive_search"] }
```

**Restart the server afterwards** — principals are read at startup.

Read names from `GET /v1/tools` rather than guessing: a name that does not
exist is a `400 Unknown tool`, and the two layers are independent. A server's
`tool_filters` decides which tools *exist*; `allowed_tools` decides who may
call them. Filtering a tool out means no grant can reach it.

Revoking an exact name does **not** override a glob that still covers it —
`grant-tools.sh` warns when that happens, since believing you removed access
when you did not is the expensive mistake.

A glob is the practical unit — adding a server should not mean re-enumerating
every principal's tool list. `"*"` grants everything; treat it as an operator
principal, not a shared key. `GET /v1/tools` shows `permitted: false` next to
`dangerous: true`, so a caller can see what to ask an operator for.

### Design notes

- **The description is the interface.** It is the only documentation the model
  gets, and a vague one produces wrong calls more often than a bad schema.
  Schemas themselves are generated from type hints and docstrings.
- **A tool never raises at the model.** A blocked address, bad arguments, a
  name that does not exist — all come back as error *results* the model reads
  and recovers from. Raising would abort the turn and discard the loop's work.
- **The loop is capped** (`max_tool_iterations`, default 5). Strands treats an
  omitted cap as *no limit*, which is how an unbounded loop once looked like a
  stuck request.
- **Prefixes prevent collisions.** Two servers both exposing `search` would
  shadow each other; the second is skipped and the operator told to set a
  distinct `prefix`.

### Why `http_request` refuses internal addresses

A model-chosen URL is a server-side request forgery engine. Anything the
*server* can reach becomes reachable: cloud instance metadata at
`169.254.169.254`, internal admin panels, a private-subnet database — or this
harness on loopback, where the tool could read other principals' sessions.

Validation runs against the **resolved IPs**, not the hostname, because a DNS
name can point anywhere (`localhost.example.com` → `127.0.0.1` is a real
technique), and **every redirect hop is revalidated**, because a redirect
target is chosen by the remote server. `HARNESS_TOOL_HTTP_ALLOWED_HOSTS`
narrows it further.

## How context is preserved

A session owns **one canonical transcript**. On every turn the whole
transcript is replayed to whichever provider the model id resolves to. What
survives a provider switch, exactly:

| | Same provider | Switched provider |
|---|---|---|
| User turns, including images | ✅ | ✅ |
| Assistant text | ✅ | ✅ |
| Anthropic thinking blocks + signatures | ✅ replayed verbatim | ❌ omitted, switch reported in `adjustments` |

Each assistant turn is stored twice: once canonically, and once as the
provider's own payload. An adapter serving a turn it produced earlier replays
that native payload verbatim — which is what keeps Anthropic thinking blocks
and their signatures valid. On a switch the other vendor would reject or
ignore those blocks, so the canonical text is sent instead. Neither
`provider` nor `native` is ever exposed over HTTP.

## Adding a provider

1. Add its models to `core/registry.py` with their capability flags.
2. Write an adapter implementing `providers/base.py:LLMProvider` — four
   members: `name`, `available()`, `converse()`, `stream()`.
3. Register it in `core/service.py:build_providers`.

Nothing else changes. The API layer, the session store, and the canonical
types are all provider-agnostic.

## Layout

```
src/morpheus/
  __main__.py                  `serve` and `mint-key` subcommands
  config.py                    Settings from env / .env
  errors.py                    Error hierarchy -> status, codes, sanitization
  auth/principals.py           Inbound identities, key hashing, allowlists
  core/types.py                Canonical wire + domain model
  core/registry.py             Routing table and capability table
  core/service.py              Session context, routing, authorization, tool loop
  harness/mcp.py               MCP server config, lifecycle, tool discovery
  harness/tools.py             The tool catalog: builtin + MCP, with authorization
  tools/guarded.py             get_current_time, http_request implementations
  tools/net.py                 SSRF guards for outbound tool requests
  providers/base.py            The adapter contract + shared translation
  providers/credentials.py     Outbound credential-chain detection
  providers/anthropic_provider.py
  providers/openai_provider.py
  sessions/base.py             Store protocol (sessions carry an owner)
  sessions/memory.py           In-process implementation
  api/security.py              The inbound-auth dependency
  api/{app,routes,deps}.py     FastAPI surface
```

## Known limitations

These are design boundaries of this version, not bugs:

- **Sessions are in-process.** Run one worker. A second worker would not see
  sessions the first created. Fix: implement `SessionStore` over Redis and
  swap it in `api/deps.py:build_service` — nothing else changes.
- **Principals are static.** Loaded once at startup from a file; adding or
  revoking a key needs a restart, and there is no enrolment endpoint. Minting
  a credential is meant to be an operator action, not a self-service one.
- **No spend limits.** `max_tokens_per_turn` bounds a single turn, not a
  principal's daily cost. `SpendLimitExceededError` exists in the error
  hierarchy but nothing accumulates usage against it yet — the `usage` numbers
  on every response are the raw material for it.
- **No transport security of its own.** Inbound keys are bearer tokens, so
  they are only as safe as the channel. Terminate TLS in front of this, and
  keep `HARNESS_HOST` on loopback otherwise.
- **No approval gate on tool calls.** A granted tool runs without asking. For
  anything with real side effects, a human-in-the-loop confirmation step
  belongs between the model's request and execution.
- **No context-window accounting.** A transcript that outgrows the model's
  window fails at the provider. `HARNESS_SESSION_MAX_TURNS` bounds growth
  bluntly by dropping the oldest turns; token-aware trimming, or the
  providers' own compaction, would be the real fix.
- **Sampling is dropped for every Anthropic model.** The frontier models
  reject `temperature`/`top_p`/`top_k` with a 400, and `anthropic` 1.x removed
  the parameters from `messages.create` entirely, so there is no first-class
  way to send them even to an older model that would accept them.
- **OpenAI reasoning summaries are unavailable.** The adapter uses
  `chat.completions`, which does not return them. Reasoning tokens are still
  billed and are reported in `usage.reasoning_tokens`. Moving that adapter to
  the Responses API is what would surface the summaries and let reasoning
  state persist across OpenAI turns.
- **Streaming persists partial turns.** If a client disconnects mid-stream,
  whatever was generated is still committed to the session — the alternative
  is a session that has silently forgotten something the user already saw.

## Tests

```bash
.venv/bin/python -m pytest -q      # 151 tests, no credentials, no network
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Adapter translation is tested directly against the installed SDK signatures:
`test_providers.py` asserts that every keyword the harness builds is actually
accepted by `anthropic.messages.create` / `openai.chat.completions.create`,
so an SDK upgrade that moves a parameter fails the suite instead of failing in
production.
