# Getting started

Run Morpheus locally and have a first conversation — including a two-provider session — in a
few minutes.

## What you need

- **Python 3.11+**
- **An outbound credential** for at least one provider — an API key is the simplest option
  (Anthropic and/or OpenAI). Health checks, the model catalog, and the auth checks work with
  **no** provider credential at all, so you can verify the wiring before spending anything.
- **An inbound credential** — a harness-issued API key, minted below. The service refuses to
  start without inbound auth configured (or an explicit anonymous flag).

## Install and configure

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env

# Outbound: how the harness reaches the providers.
export ANTHROPIC_API_KEY=sk-ant-...

# Inbound: who may call the harness.
.venv/bin/python -m morpheus mint-key --id me > /tmp/minted
# paste the printed entry into ./principals.json, then:
export HARNESS_PRINCIPALS_FILE=./principals.json
```

`mint-key` prints the key **once** (it is not recoverable) plus the JSON entry to paste into
the principals file. The key itself is never stored — only its SHA-256.

## Run it

```bash
.venv/bin/python -m morpheus
```

- Chat UI: `http://127.0.0.1:8080/app/` (and `/` redirects there) — sign in with the minted key.
- Interactive API docs (FastAPI): `http://127.0.0.1:8080/docs`.

Then exercise the whole thing from a second terminal — health, the filtered catalog, a
two-provider session, the stored transcript, and the auth checks:

```bash
scripts/demo.sh
```

Steps 1, 2, and 6 of that script need no provider credential, so it is also a useful wiring
check before you spend anything.

For a throwaway run, `HARNESS_ALLOW_ANONYMOUS=true` skips inbound auth entirely — every caller
shares one `anonymous` principal (and can read every anonymous session). Localhost only.

## First conversation

Send only the new turn each time; the stored transcript is prepended for you.

Turn 1 — Anthropic (no `session_id`, so one is created and returned):

```bash
curl -s localhost:8080/v1/converse \
  -H "authorization: Bearer $HARNESS_KEY" \
  -H 'content-type: application/json' -d '{
  "model": "claude-opus-5",
  "messages": [{"role": "user", "content": [{"type": "text", "text": "My name is Ramesh."}]}]
}'
```

Turn 2 — OpenAI, **same session**. The second provider sees turn 1:

```bash
curl -s localhost:8080/v1/converse \
  -H "authorization: Bearer $HARNESS_KEY" \
  -H 'content-type: application/json' -d '{
  "model": "gpt-5",
  "session_id": "sess_ab12...",
  "messages": [{"role": "user", "content": [{"type": "text", "text": "What is my name?"}]}]
}'
```

The response is the same shape whichever provider served it, including `provider`, `usage`
(input/output/cache/reasoning tokens), `latency_ms`, and `adjustments`.

## Run the tests

The suite runs with **no credentials and no network**:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Adapter translation is tested directly against the installed SDK signatures, so an SDK upgrade
that moves a parameter fails the suite instead of failing in production.

## Next

- [Capabilities](features.md) — the full API and feature tour
- [Operating it](operations.md) — keys, principals, tools, and deployment
