#!/usr/bin/env bash
# End-to-end demo: one session, two providers, context carried across both.
#
#   scripts/demo.sh [base-url]
#
# Reads the inbound key from .harness-key. Requires the server to be running
# and at least one provider credential configured.
set -uo pipefail

BASE="${1:-http://127.0.0.1:8080}"
KEY="$(cat "$(dirname "$0")/../.harness-key" 2>/dev/null || true)"
if [ -z "$KEY" ]; then
  echo "No .harness-key found. Mint one:" >&2
  echo "  python -m morpheus mint-key --id local-dev" >&2
  exit 1
fi

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

call() { # call <model> <text> [session_id]
  local body
  body=$(jq -n --arg m "$1" --arg t "$2" --arg s "${3:-}" '
    {model: $m, messages: [{role: "user", content: [{type: "text", text: $t}]}]}
    + (if $s == "" then {} else {session_id: $s} end)')
  curl -sS -X POST "$BASE/v1/converse" \
    -H "authorization: Bearer $KEY" \
    -H 'content-type: application/json' \
    -d "$body"
}

say "1. Health — which credential did each provider resolve?"
curl -sS "$BASE/v1/health" | jq .

say "2. Models this key may use"
curl -sS "$BASE/v1/models" -H "authorization: Bearer $KEY" \
  | jq -r '.models[] | "  \(.id)  (\(.provider))  available=\(.available)"'

# Pick one model per provider that can actually serve. "available" alone is
# not enough: the Anthropic adapter reports itself available even with no
# credential found, because it resolves lazily. Use health's
# credential_detected as the gate so the demo does not pick a model that is
# certain to fail.
HEALTH=$(curl -sS "$BASE/v1/health")
MODELS=$(curl -sS "$BASE/v1/models" -H "authorization: Bearer $KEY")
usable() { echo "$HEALTH" | jq -r --arg p "$1" '.providers[$p] | (.available and .credential_detected)'; }
pick() { echo "$MODELS" | jq -r --arg p "$1" '[.models[] | select(.provider==$p)][0].id // empty'; }

A=""; O=""
[ "$(usable anthropic)" = "true" ] && A=$(pick anthropic)
[ "$(usable openai)" = "true" ] && O=$(pick openai)
FIRST="${A:-$O}"
SECOND="${O:-$A}"

if [ -z "$FIRST" ]; then
  echo
  echo "No provider has a usable credential yet, so there is nothing to call." >&2
  echo "$HEALTH" | jq -r '.providers | to_entries[] | "  \(.key): \(.value.note)"' >&2
  echo >&2
  echo "Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY, restart the server, and" >&2
  echo "run this again. Steps 1, 2 and 6 above work without any credential." >&2
  exit 1
fi

show() { # print a turn, or its error
  jq 'if .error then {error: .error} else
    {session_id, provider, model, native_model,
     text: (.output.message.content | map(select(.type=="text").text) | join("")),
     usage, latency_ms, adjustments} end'
}

say "3. Turn one on $FIRST (new session)"
R1=$(call "$FIRST" "My name is Ramesh and my favourite number is 41. Reply in one short sentence.")
echo "$R1" | show
SID=$(echo "$R1" | jq -r '.session_id // empty')
if [ -z "$SID" ]; then
  echo >&2
  echo "First turn failed — see the error above. A 502 provider_auth_failed" >&2
  echo "means the key was rejected; a 503 means none was found." >&2
  exit 1
fi

if [ "$FIRST" != "$SECOND" ]; then
  say "4. Turn two on $SECOND — same session, different provider"
  echo "   (if it answers correctly, context crossed the provider boundary)"
  call "$SECOND" "What is my name and my favourite number?" "$SID" | show
else
  say "4. Only one provider is configured — staying on $FIRST"
  call "$FIRST" "What is my name and my favourite number?" "$SID" | show
fi

say "5. The stored transcript"
curl -sS "$BASE/v1/sessions/$SID?include_messages=true" -H "authorization: Bearer $KEY" \
  | jq '{owner, turn_count, last_provider, messages: [.messages[] | {role, text: (.content | map(select(.type=="text").text) | join(""))}]}'

say "6. Auth is enforced"
printf '  no key:       HTTP %s\n' "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/v1/sessions")"
printf '  wrong key:    HTTP %s\n' "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/v1/sessions" -H 'authorization: Bearer mh_wrong')"
printf '  valid key:    HTTP %s\n' "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/v1/sessions" -H "authorization: Bearer $KEY")"
printf "  other's session: HTTP %s (404 = ownership enforced)\n" \
  "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/v1/sessions/sess_someone_else" -H "authorization: Bearer $KEY")"

say "Done. Session id: $SID"
