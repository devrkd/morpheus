#!/usr/bin/env bash
# Does my GitHub token work with the remote MCP server?
#
#   scripts/check-github-mcp.sh              # uses $GITHUB_TOKEN
#   scripts/check-github-mcp.sh ghp_xxx      # or pass one
#
# Talks to GitHub directly, not through the harness, so it isolates the
# question of whether the token is accepted at all.
set -uo pipefail

TOKEN="${1:-${GITHUB_TOKEN:-}}"
ENDPOINT="${GITHUB_MCP_URL:-https://api.githubcopilot.com/mcp/}"
export NO_PROXY='*'

if [ -z "$TOKEN" ]; then
  echo "No token. Pass one, or export GITHUB_TOKEN." >&2
  exit 2
fi

case "$TOKEN" in
  ghp_*)        KIND="classic PAT — the server will hide tools your scopes don't cover" ;;
  github_pat_*) KIND="fine-grained PAT — no scope filtering; all tools listed, API enforces per call" ;;
  gho_*|ghu_*)  KIND="OAuth token" ;;
  *)            KIND="unrecognised prefix — is this a GitHub token?" ;;
esac
printf 'token:    %s… (%s chars)\n' "${TOKEN:0:8}" "${#TOKEN}"
printf 'kind:     %s\n' "$KIND"
printf 'endpoint: %s\n\n' "$ENDPOINT"

BODY='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"model-harness-check","version":"1"}}}'

TMPBASE="${TMPDIR:-/tmp}"
HEADERS=$(mktemp "$TMPBASE/mcpcheck-h.XXXXXX")
RESP=$(mktemp "$TMPBASE/mcpcheck-r.XXXXXX")
trap 'rm -f "$HEADERS" "$RESP"' EXIT

CODE=$(curl -sS --max-time 20 -o "$RESP" -D "$HEADERS" -w '%{http_code}' \
  -X POST "$ENDPOINT" \
  -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -H 'mcp-protocol-version: 2025-06-18' \
  -d "$BODY" 2>&1) || { echo "Could not reach $ENDPOINT" >&2; exit 1; }

echo "HTTP $CODE"
case "$CODE" in
  200)
    echo "  ✅ the token is accepted — a PAT is all you need"
    SESSION=$(grep -i '^mcp-session-id:' "$HEADERS" | tr -d '\r' | cut -d' ' -f2- || true)
    SERVER=$(tr -d '\r' < "$RESP" | sed -n 's/^data: //p' | tail -1)
    [ -z "$SERVER" ] && SERVER=$(cat "$RESP")
    echo "$SERVER" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
info = d.get("result", {}).get("serverInfo", {})
if info:
    print(f"  server: {info.get(\"name\")} {info.get(\"version\", \"\")}")
' 2>/dev/null || true

    if [ -n "$SESSION" ]; then
      curl -sS --max-time 20 -o /dev/null -X POST "$ENDPOINT" \
        -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
        -H 'accept: application/json, text/event-stream' \
        -H "mcp-session-id: $SESSION" -H 'mcp-protocol-version: 2025-06-18' \
        -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' || true

      COUNT=$(curl -sS --max-time 25 -X POST "$ENDPOINT" \
        -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
        -H 'accept: application/json, text/event-stream' \
        -H "mcp-session-id: $SESSION" -H 'mcp-protocol-version: 2025-06-18' \
        -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
        | tr -d '\r' | sed -n 's/^data: //p' | tail -1 \
        | python3 -c '
import json, sys
try:
    tools = json.load(sys.stdin)["result"]["tools"]
except Exception:
    print(""); sys.exit(0)
print(len(tools))
names = sorted(t["name"] for t in tools)
print("  first few: " + ", ".join(names[:8]), file=sys.stderr)
' 2>"$TMPBASE/mcp_names") || true
      [ -n "${COUNT:-}" ] && echo "  tools exposed to this token: $COUNT"
      cat "$TMPBASE/mcp_names" 2>/dev/null; rm -f "$TMPBASE/mcp_names"
    fi
    echo
    echo "  Use it as-is:"
    echo "    \"headers\": { \"Authorization\": \"Bearer \${GITHUB_TOKEN}\" }"
    ;;
  401)
    echo "  ❌ rejected. The token is invalid, expired, or PATs are disabled for your account."
    echo "     If you are an Enterprise Managed User, PATs are off unless an"
    echo "     enterprise admin enables them — that is the usual cause here."
    ;;
  403)
    echo "  ❌ authenticated but not entitled. Usually a missing Copilot plan"
    echo "     (any plan works, including Free) or an org policy on PATs."
    ;;
  404) echo "  ❌ endpoint not found — check GITHUB_MCP_URL." ;;
  *)   echo "  ❓ unexpected. Body:"; head -c 400 "$RESP" | sed 's/^/     /' ;;
esac

if [ "$CODE" != "200" ]; then
  echo
  echo "  Fallback that avoids the remote endpoint and needs no Copilot plan:"
  echo "  run GitHub's server locally (you have Docker), in mcp.json:"
  cat <<'JSON'
     "github": {
       "command": "docker",
       "args": ["run", "-i", "--rm",
                "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server"],
       "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}" },
       "prefix": "github"
     }
JSON
fi
