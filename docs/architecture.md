# Architecture

How Morpheus works internally, at a functional level. No code walkthrough — component
responsibilities, the request lifecycle, and the design boundaries that shape deployment.

## At a glance

Morpheus is a **harness layer** on top of the [Strands](https://github.com/strands-agents/strands)
agent framework, exposed through FastAPI:

| Component | Responsibility |
|---|---|
| **API layer** (`api/`) | The FastAPI surface: routing, request validation, inbound-auth dependency, and the single-file web chat client |
| **Model registry** (`core/`) | The canonical wire types and the **static routing + capability table** — one harness model id maps to exactly one provider and its capabilities |
| **Harness runner** (`harness/`) | The agent loop: routing, transcript attachment, parameter translation, authorization, the tool loop, and error translation — built on Strands' `Agent` |
| **Model factory** (`harness/models.py`) | Builds provider connections from Strands' bundled Anthropic/OpenAI adapters and Morpheus' own credential detection |
| **Authentication** (`auth/`) | Inbound principals: key minting, SHA-256 storage, allowlists |
| **Tools** (`harness/tools.py`, `tools/`) | The tool catalog (built-in + MCP), permission checks, and SSRF guards on outbound fetches |
| **MCP registry** (`harness/mcp.py`) | MCP server configuration, lifecycle, and tool discovery |
| **Session store** | File-based transcripts + an ownership index, via Strands' session manager |
| **Provider credentials** (`providers/`) | Outbound credential-chain detection for Anthropic and OpenAI |

The framework supplies the agent loop, the provider SDK adapters, the MCP client, and file
persistence; Morpheus contributes everything that makes the result a *managed service*: the
static catalog, authentication and authorization, parameter translation with `adjustments`
reporting, guarded tools, and the unified Converse API.

## Request lifecycle

```
client ──POST /v1/converse──► API layer
                                1. authenticate: Bearer mh_… → principal
                                2. validate request against canonical schema
                                3. route: model id → ModelSpec (provider + capabilities)
                                4. authorize: model in principal's allowlist?
                                5. load session transcript, prepend to new turn
                                6. translate request for the target provider
                                   (drop/clamp what the model can't take; record it)
                                7. call provider (via Strands adapter)
                                8. run tool loop if the model asks for tools
                                   (existence + permission check per tool, capped)
                                9. store the new transcript
                               10. respond: canonical shape + usage + adjustments
```

Streaming follows the same lifecycle but emits server-sent events and delivers the session id
in a response header before the first byte.

## The model registry

Routing is a **static table**, not runtime discovery. Each entry records:

- the harness **model id** clients send, its **aliases**, and the provider it resolves to;
- the **native id** used with that provider;
- capabilities: context window, max output tokens, thinking style, whether it accepts images,
  sampling parameters, stop sequences, and its effort levels.

This design has two consequences worth knowing:

- Routing is answerable without a network call — and a model the harness has never been told
  about fails fast with a clear error instead of being guessed at.
- Allowlists are enforced against the **canonical** id, so an alias cannot slip past a list
  that omits its model.

## Sessions and context preservation

A session owns **one canonical transcript**. On every turn the whole transcript is replayed to
whichever provider the model id resolves to. What survives a provider switch, exactly:

| | Same provider | Switched provider |
|---|---|---|
| User turns, including images | ✅ | ✅ |
| Assistant text | ✅ | ✅ |
| Provider-specific reasoning blocks | ✅ replayed verbatim | ❌ omitted, and the switch is reported in `adjustments` |

Each assistant turn is stored twice: once canonically, and once as the provider's own payload.
An adapter serving a turn it produced earlier replays that native payload verbatim — which is
what keeps provider-specific blocks valid. On a switch, the other vendor would reject or ignore
those blocks, so the canonical text is sent instead.

Sessions live in the service process (file-backed, under the configured session directory) with
a TTL and a hard cap on stored turns. **One consequence:** run a single worker. See
[Known limitations](#known-limitations).

## Authentication and authorization

Two independent directions, kept deliberately separate:

**Inbound (callers → harness).** A caller presents a harness-issued key as a Bearer token.
Keys are minted by an operator and stored only as a SHA-256 digest — 256 bits of
machine-generated randomness, so there is no dictionary to attack and nothing a slow KDF would
buy. A principal carries a model allowlist, a tool allowlist, a per-turn token ceiling, and a
disabled flag. Startup fails closed when inbound auth is unconfigured.

**Outbound (harness → providers).** Each provider's SDK walks its own credential chain — API
key, `ant auth login` profile, or workload identity federation for Anthropic; an API key or
workload identity for OpenAI. Only explicitly configured values are passed to the SDK;
everything else is left for the chain. `/v1/health` reports which *source* each provider
resolved — a source name, never a value.

## The tool loop

Tools come from the built-in catalog and from configured MCP servers. MCP servers are loaded
**once at startup** (each is a session; per-request handshakes would dominate short turns) and
fail independently — one misconfigured server never takes the API down.

Three properties define the loop:

1. **Secrets never enter the config or the model.** `${VAR}` in the MCP config is interpolated
   from the environment at load time; tokens stay in the service and are attached when calling
   the server.
2. **A tool never raises at the model.** A blocked address, bad arguments, an unknown name —
   all come back as error *results* the model reads and recovers from.
3. **The loop is capped** (default 5 iterations), and usage is summed across the whole loop —
   a tool turn costs every request in it, not just the last.

Permissions are two independent gates: the tool must **exist** (a server's filters decide
which tools exist) and be **granted** to the principal (by exact name or glob). Every MCP tool
is considered dangerous and is ungranted by default — even in anonymous mode.

## Error handling

Morpheus has a single error hierarchy that decides what each error becomes: HTTP status, a
stable `code`, whether it is retryable, and what is safe to tell the caller. Caller errors
explain themselves; provider-internal errors are sanitized and logged against an `error_id`.
Invalid and disabled keys return byte-identical responses.

## Known limitations

These are **design boundaries** of the current version, not bugs:

- **Sessions are in-process.** Run one worker. A second worker would not see sessions the first
  created. Multi-instance deployment means moving transcripts and the ownership index to a
  shared backend.
- **Principals are static.** Loaded once at startup from a file; adding or revoking a key
  needs a restart, and there is no enrolment endpoint — minting a credential is an operator
  action by design.
- **No spend limits.** `max_tokens_per_turn` bounds a single turn, not a principal's daily
  cost. The per-turn `usage` numbers are the raw material for adding this.
- **No transport security of its own.** Inbound keys are bearer tokens, only as safe as the
  channel. Terminate TLS in front of the service, and keep it on loopback otherwise.
- **No approval gate on tool calls.** A granted tool runs without asking. For anything with
  real side effects, a human-in-the-loop confirmation belongs between the model's request and
  execution.
- **No context-window accounting.** A transcript that outgrows the model's window fails at the
  provider. The turn cap bounds growth bluntly by dropping oldest turns.
- **Sampling parameters are dropped for Anthropic models** (frontier models reject them, and
  the current SDK removed them) — always reported in `adjustments`.
- **OpenAI reasoning summaries are unavailable** through the current adapter; reasoning tokens
  are still billed and reported.
- **Streaming persists partial turns.** If a client disconnects mid-stream, what was generated
  is still committed — the alternative is a session that silently forgot something the user
  already saw.
