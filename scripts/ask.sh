#!/usr/bin/env bash
# Send one prompt and print the reply.
#
#   scripts/ask.sh "your prompt"                       # new session
#   scripts/ask.sh "your prompt" gpt-5                 # pick a model
#   scripts/ask.sh "follow up" claude-sonnet-5 sess_ab # continue a session
#
#   -c, --continue      continue the most recent session (no id to copy)
#   SESSION=sess_ab     continue that session, without naming a model
#   VERBOSE=1           print the full JSON response
#   WITH_TIME=1         append the wall clock to the prompt
#
# Prints the session id to stderr so it stays out of piped output.
set -uo pipefail

CONTINUE=""
if [ "${1:-}" = "-c" ] || [ "${1:-}" = "--continue" ]; then
  CONTINUE=1
  shift
fi

PROMPT="${1:?usage: scripts/ask.sh [-c] \"prompt\" [model] [session_id]}"
MODEL="${2:-claude-sonnet-5}"
BASE="${HARNESS_URL:-http://127.0.0.1:8080}"

cd "$(dirname "$0")/.."

# Session id, in precedence order: 3rd argument, then $SESSION, then the last
# session this script used when -c was passed.
SESSION="${3:-${SESSION:-}}"
LAST_FILE=".harness-session"
if [ -z "$SESSION" ] && [ -n "$CONTINUE" ]; then
  SESSION="$(cat "$LAST_FILE" 2>/dev/null || true)"
  [ -z "$SESSION" ] && echo "note: no previous session recorded; starting a new one" >&2
fi
KEY="$(cat .harness-key 2>/dev/null || true)"
[ -z "$KEY" ] && { echo "No .harness-key — run: python -m model_harness mint-key --id local-dev" >&2; exit 1; }

# A model has no clock. To let it answer a "what time is it" style question,
# the time has to travel in the request. It is appended to the USER turn, not
# the system prompt, on purpose: prompt caching is a prefix match and `system`
# renders before `messages`, so a timestamp up there would change the cached
# prefix on every single turn and drive the cache hit rate to zero. Volatile
# content belongs after the last cache breakpoint.
TEXT="$PROMPT"
if [ -n "${WITH_TIME:-}" ]; then
  TEXT="$PROMPT

(For reference, the current local time is $(date '+%Y-%m-%d %H:%M:%S %Z').)"
fi

BODY=$(jq -n --arg m "$MODEL" --arg t "$TEXT" --arg s "$SESSION" '
  {model: $m, messages: [{role: "user", content: [{type: "text", text: $t}]}]}
  + (if $s == "" then {} else {session_id: $s} end)')

RESP=$(curl -sS --noproxy '*' "$BASE/v1/converse" \
  -H "authorization: Bearer $KEY" \
  -H 'content-type: application/json' -d "$BODY") || {
    echo "Could not reach $BASE — is the server running?" >&2; exit 1; }

if [ -n "${VERBOSE:-}" ]; then
  echo "$RESP" | jq .
  exit 0
fi

# An error payload goes to stderr and exits non-zero, so this composes in a pipeline.
if echo "$RESP" | jq -e '.error' >/dev/null 2>&1; then
  echo "$RESP" | jq -r '"error \(.error.code): \(.error.message)  [\(.error.error_id)]"' >&2
  exit 1
fi

echo "$RESP" | jq -r '.output.message.content | map(select(.type=="text").text) | join("")'

# Remember it so `-c` can continue without copying an id around.
echo "$RESP" | jq -r .session_id > "$LAST_FILE"

{
  echo
  echo "  session: $(echo "$RESP" | jq -r .session_id)   provider: $(echo "$RESP" | jq -r .provider)   model: $(echo "$RESP" | jq -r .native_model)"
  echo "  tokens:  in=$(echo "$RESP" | jq -r .usage.input_tokens) out=$(echo "$RESP" | jq -r .usage.output_tokens) cache_read=$(echo "$RESP" | jq -r .usage.cache_read_tokens)   ${_L:-}$(echo "$RESP" | jq -r .latency_ms)ms"
  ADJ=$(echo "$RESP" | jq -r '.adjustments | if length == 0 then "" else "  adjustments: " + join("; ") end')
  if [ -n "$ADJ" ]; then echo "$ADJ"; fi
} >&2

# Explicit: without this the exit status is that of the last test above, so a
# successful call with no adjustments would report failure and break `&&`.
exit 0
