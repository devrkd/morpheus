#!/usr/bin/env bash
# Exercise the tool loop against a real model.
#
#   scripts/tools-demo.sh [base-url]
set -uo pipefail
BASE="${1:-http://127.0.0.1:8080}"
cd "$(dirname "$0")/.."
KEY="$(cat .harness-key)"
export NO_PROXY='*'

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

ask() { # ask <prompt> <tool>...
  local prompt="$1"; shift
  local tools; tools=$(printf '%s\n' "$@" | jq -R . | jq -sc .)
  curl -sS --noproxy '*' "$BASE/v1/converse" \
    -H "authorization: Bearer $KEY" -H 'content-type: application/json' \
    -d "$(jq -n --arg m "${MODEL:-claude-sonnet-5}" --arg t "$prompt" --argjson tl "$tools" \
      '{model: $m, tools: $tl, messages: [{role:"user", content:[{type:"text", text:$t}]}]}')"
}

show() {
  jq -r 'if .error then "  ERROR \(.error.code): \(.error.message)"
   else
     "  answer: " + (.output.message.content | map(select(.type=="text").text) | join("")) + "\n" +
     "  iterations: \(.iterations)   tokens: in=\(.usage.input_tokens) out=\(.usage.output_tokens)\n" +
     "  tools run: " + (if (.tool_calls|length) == 0 then "(none)"
        else (.tool_calls | map("\(.name)(\(.input|tostring)) -> \(if .is_error then "ERROR " else "" end)\(.output | .[0:90] | gsub("\n";" "))") | join("\n             ")) end)
   end'
}

say "Tools this key may use"
curl -sS --noproxy '*' "$BASE/v1/tools" -H "authorization: Bearer $KEY" \
  | jq -r '.tools[] | "  \(.name)  dangerous=\(.dangerous)  permitted=\(.permitted)"'

say "1. The clock — the question that failed before"
ask "What is the current date and time in Amsterdam?" get_current_time | show

say "2. Arithmetic via code execution"
ask "What is 2**200 divided by 3, as an exact integer remainder? Use code." run_python | show

say "3. A live HTTP call"
ask "Fetch https://api.github.com/zen and tell me exactly what it said." http_request | show

say "4. SSRF guard: the model is told no, and recovers"
ask "Fetch http://169.254.169.254/latest/meta-data/ and report what you find." http_request | show

say "5. Two tools in one turn"
ask "What is the current UTC time, and what is the 30th Fibonacci number? Use tools for both." get_current_time run_python | show
