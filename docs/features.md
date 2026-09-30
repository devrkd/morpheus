# Capabilities

A tour of what Morpheus does, from the caller's point of view. Everything here is available
through one HTTP API and a small built-in chat UI.

## One API, many models

Morpheus exposes a single conversation endpoint, no matter which vendor answers:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/converse` | Send a turn, get the whole response |
| `POST` | `/v1/converse-stream` | Same, streamed as server-sent events |
| `GET` | `/v1/models` | The model catalog, filtered to what *you* may use |
| `GET` | `/v1/tools` | Tools available to you, with schemas and permissions |
| `GET` | `/v1/whoami` | Your identity and limits — the cheapest way to validate a key |
| `GET` | `/v1/sessions` | Your session ids, or summaries with `?detail=true` |
| `GET` | `/v1/sessions/{id}` | A session's metadata, or its transcript with `?include_messages=true` |
| `DELETE` | `/v1/sessions/{id}` | Forget a session |
| `GET` | `/v1/health` | **Open.** Liveness plus per-provider credential status |

The response shape is identical whichever provider answered. A caller never parses
vendor-specific fields: the provider is a reported fact (and can change mid-conversation), not
a client concern.

## Conversations that survive anything

- **Only the new turn is uploaded.** The stored transcript is prepended for you, so client code
  stays trivial.
- **The transcript is canonical and server-side.** A session stores one transcript; the whole
  history is replayed to whichever provider handles each turn. Open the conversation from
  another tab, machine, or client — it is still there.
- **Sessions belong to a caller.** Every session records its creator. Another principal reading,
  deleting, or appending to it gets `404`, not `403` — a `403` would confirm the id exists.
- **Provider switches preserve meaning.** If you ask for `gpt-5` in a session started on
  `claude-opus-5`, the new model sees the full user/assistant history. (See
  [Architecture → Sessions](architecture.md#sessions-and-context-preservation) for the exact
  preservation rules.)

## Streaming

`/v1/converse-stream` streams the response as server-sent events with a small canonical
vocabulary — `message_start`, `content_block_start`, `content_delta`, `content_block_stop`,
`message_stop`, `error` — then a final `[DONE]`. The session id arrives in an `X-Session-Id`
header before the first body byte. A mid-stream failure is delivered as an `error` event (the
HTTP status is already sent once a stream starts); treat it as terminal.

## The model catalog

`GET /v1/models` returns the models a caller may use, each with its capabilities: context
window, maximum output tokens, whether it accepts images, its reasoning style, effort levels,
and aliases. Two properties matter:

- The catalog is **filtered to your allowlist** — models you may not use are omitted, not
  listed as forbidden.
- The catalog is **static, not discovered**: routing is answerable without a network call, and
  a model the harness has never been told about fails fast with a clear error.

## `adjustments`: nothing is silently altered

Provider models differ in what parameters they accept. When a caller sends a parameter the
target model cannot take verbatim, Morpheus never forwards it (which would earn a 400) and
never hides that it dropped it — the response carries an `adjustments` list, for example:

```json
"adjustments": [
  "dropped temperature, top_p: not accepted by claude-opus-5",
  "effort 'max' clamped to 'high' (highest supported by gpt-5)",
  "max_tokens 100000 lowered to 16384, the output ceiling for gpt-4o"
]
```

What the client sent, what was changed, and why — visible in every response.

## Tools: from talking to doing

Without tools, a model can only talk — no clock, no network, no computation. Morpheus gives
models tools from two places:

**Built in** (two, kept for specific reasons):

| Tool | What it does |
|---|---|
| `get_current_time` | The most common thing a model is asked and cannot know |
| `http_request` | Outbound HTTP fetch, with server-side request-forgery protection: internal, loopback, private, and cloud-metadata addresses are refused — validated against resolved IPs and re-validated on every redirect hop. Writes (`POST`/`PUT`/`PATCH`/`DELETE`) are off by default. |

**From MCP servers** — any Model Context Protocol server you configure: GitHub, Google Drive,
an internal service, or anything else that speaks MCP. Configuration uses the same
`mcpServers` JSON format as Claude Desktop, so an existing config works unchanged.

Tools are requested per turn; the response reports what ran, how many provider round trips the
tool loop took, and usage summed across the whole loop. Permission rules:

- A tool must both **exist** and be **granted to the caller** — two independent gates.
- **Every MCP tool counts as dangerous** (it runs code you did not write against a system you
  do not control, with credentials the service holds), so none is granted by default — even in
  anonymous mode.
- Grants are by exact name or glob, per principal, in the principals file.

Details and grant commands: [Operating it → Tools](operations.md#tools-and-permissions).

## Security model

**Inbound** — callers authenticate with a harness-issued API key (`Bearer mh_…`). Keys are
minted by an operator and stored only as a SHA-256 digest; a lost key is not recoverable. Each
principal (a team, a service, a person) carries:

- a **model allowlist** — enforced against the canonical model id, so an alias cannot slip past
  a list that omits its model;
- **session ownership** — sessions are per-principal;
- a **per-turn token ceiling** — reported in `adjustments` when it binds;
- **revocation** — `disabled: true` or deleting the entry takes effect on the next restart,
  without touching any provider credential.

Startup **fails closed**: with neither a principals file nor an explicit anonymous-mode flag,
the service refuses to start rather than run as an open proxy for credentials that cost money.

**Outbound** — each provider's SDK resolves its own credentials (API key, `ant auth login`
profile, or workload identity federation), so no provider key has to be committed to config.

A full walkthrough: [Architecture → Authentication](architecture.md#authentication-and-authorization).

## Error handling designed for humans and ops

- Errors about **the caller's own request** explain themselves.
- Errors originating **inside a provider** return a generic message plus an `error_id` — the
  full detail is logged against that id, so a support report quoting it leads straight to the
  context, without leaking provider internals to callers.
- An invalid key and a disabled key return **byte-identical responses**, so the API cannot be
  used to enumerate valid keys.

## Observability

- **`/v1/health`** (the one unauthenticated route): liveness, per-provider availability, and
  which credential *source* each provider resolved — a source name, never a value.
- **`/v1/whoami`**: the calling principal's identity and limits.
- **Per-turn usage**: input, output, cache read/write, and reasoning tokens, plus latency, on
  every response — the raw material for spend accounting.

## The web client

The service serves a chat UI at `/app/`. It exists to demonstrate the properties above: opening
a session reloads it from the server; each reply shows its input-token count (watch replayed
context grow, and `cached` appear once a session hits the prompt cache); sessions are
per-principal; and `adjustments` are surfaced rather than hidden. It is served from the harness
itself — same origin, so no CORS and no second process.
