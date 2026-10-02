# Morpheus

**Morpheus** is a unified inference service: one HTTP API through which a team can use AI
models from different vendors without learning each vendor's SDK, API, or quirks.

You send one request, name a model, and Morpheus does the rest: it decides which provider
serves that model, translates the request into that provider's format, calls it, and
translates the answer back into one consistent shape. Your integration code stays the same no
matter which vendor actually answered.

## The problem it solves

Without a service like this, adopting a second AI vendor means:

- learning a second API and SDK;
- rewriting the integration code you already wrote for the first vendor;
- maintaining separate key handling, error handling, and usage tracking per vendor;
- and losing conversation context whenever you try a model from the other vendor.

Morpheus replaces all of that with **one API, one model id per request, one response shape** —
and it keeps every conversation's history on the server, so a conversation can even switch
vendors **mid-conversation** and the new model still sees the full history.

## In one paragraph

Morpheus is a Bedrock-style gateway in front of Anthropic and OpenAI. A request names a model
id — for example `claude-opus-5` or `gpt-5` — and Morpheus routes it to the matching provider,
translating requests and responses both ways. Conversations are stored server-side as canonical
transcripts, so sessions survive provider switches, page reloads, and machine changes. Around
that core it provides the controls an operator needs: API-key authentication with per-team model
allowlists and spend ceilings, tools (built-in and Model Context Protocol servers) behind
explicit permission grants, streaming, and honest reporting of every parameter it had to
adjust. It ships as a small Python service with a built-in chat UI.

## What you get

- **One API** — `POST /v1/converse` (plus a streaming variant) regardless of which provider
  answers.
- **Provider freedom** — switch models or vendors per request, or mid-conversation, without
  changing client code.
- **Server-side sessions** — the transcript lives on the server and belongs to a specific
  caller, not to a browser tab.
- **Operator controls** — minted API keys, per-principal model allowlists, per-turn token
  ceilings, and instant revocation.
- **Tools, permission-gated** — a built-in clock and a guarded HTTP fetch, plus any MCP server
  you configure; nothing dangerous is available unless explicitly granted.
- **Observability** — an open health endpoint, per-provider credential status, per-turn token
  usage, and an `adjustments` field that never silently drops a parameter.
- **A chat client included** — the service serves its own web UI, so there is no CORS setup and
  no second process to run.

## The mental model

```
                    ┌─────────────────────────────────────────────┐
 POST /v1/converse  │  Morpheus                                   │
 { model, messages, │   1. authenticate the caller (API key)      │
   session_id }     │   2. route the model id to a provider       │
        ──────────► │   3. attach the stored transcript           │──► Anthropic
                    │   4. translate request and response         │──► OpenAI
                    │   5. store the transcript, report usage     │
                    └─────────────────────────────────────────────┘
```

The caller always speaks the same language. Morpheus is the only component that knows — or
needs to know — how each provider's API differs.

## How to use these pages

| Page | What it covers |
|---|---|
| [Capabilities](features.md) | What the service can do, end to end, with the API surface |
| [Architecture](architecture.md) | How it works internally, at a functional level |
| [Getting started](getting-started.md) | Run it locally and have a first conversation |
| [Operating it](operations.md) | Keys, principals, tools, health, and deployment notes |

## Project status

Morpheus is an early-stage project (v0.1.0). It is a working service — a full test suite runs
with no credentials and no network — but several boundaries are deliberate, documented design
choices rather than bugs. See [Known limitations](architecture.md#known-limitations) before
depending on it in production.
