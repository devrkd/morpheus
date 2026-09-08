#!/usr/bin/env bash
# Grant tools to a principal in the principals file.
#
#   scripts/grant-tools.sh local-dev github_get_commit
#   scripts/grant-tools.sh local-dev 'github_*'          # a whole server
#   scripts/grant-tools.sh --revoke local-dev github_create_issue
#   scripts/grant-tools.sh --list                        # who has what
#
# Principals are read at startup, so restart the server afterwards.
set -uo pipefail
cd "$(dirname "$0")/.."

FILE="${HARNESS_PRINCIPALS_FILE:-./principals.json}"
MODE=grant
case "${1:-}" in
  --revoke) MODE=revoke; shift ;;
  --list)   MODE=list;   shift ;;
  -h|--help)
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 0 ;;
esac

[ -f "$FILE" ] || { echo "No principals file at $FILE" >&2; exit 2; }

if [ "$MODE" = list ]; then
  python3 - "$FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
for p in data.get("principals", []):
    tools = p.get("allowed_tools")
    shown = "safe tools only (default)" if tools is None else ", ".join(sorted(tools))
    print(f"  {p['id']:20} {shown}")
PY
STATUS=$?

# Without this the restart note prints even when the edit failed,
# which reads as success.
[ "$STATUS" -eq 0 ] || exit "$STATUS"
  exit 0
fi

PRINCIPAL="${1:-}"; shift || true
if [ -z "$PRINCIPAL" ] || [ "$#" -eq 0 ]; then
  echo "usage: scripts/grant-tools.sh [--revoke] <principal-id> <tool> [tool...]" >&2
  exit 2
fi

python3 - "$FILE" "$MODE" "$PRINCIPAL" "$@" <<'PY'
import json, sys
from pathlib import Path

path, mode, principal_id, *tools = sys.argv[1:]
p = Path(path)
data = json.loads(p.read_text(encoding="utf-8"))
entries = data.get("principals", [])

target = next((e for e in entries if e.get("id") == principal_id), None)
if target is None:
    ids = ", ".join(e.get("id", "?") for e in entries) or "none"
    sys.exit(f"No principal '{principal_id}' in {path}. Known: {ids}")

current = set(target.get("allowed_tools") or [])
before = sorted(current)

if mode == "grant":
    current |= set(tools)
else:
    missing = [t for t in tools if t not in current]
    current -= set(tools)
    for t in missing:
        print(f"  note: '{t}' was not in the list to begin with")

    # Revoking an exact name does nothing if a glob still covers it — the
    # quiet way to believe you have removed access when you have not.
    from fnmatch import fnmatch

    for t in tools:
        covering = [g for g in current if "*" in g and fnmatch(t, g)]
        if covering:
            print(
                f"  WARNING: '{t}' is still granted by {', '.join(sorted(covering))} — "
                "revoke that too, or replace it with exact names"
            )

target["allowed_tools"] = sorted(current)

# Written via a temp file and replaced, so an interrupted write cannot leave
# the principals file — the thing that gates every request — truncated.
tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
tmp.replace(p)

if "*" in current and mode == "grant":
    print("  note: this principal already has '*', which grants every tool")

print(f"  principal: {principal_id}")
print(f"  before:    {', '.join(before) or 'safe tools only (default)'}")
print(f"  after:     {', '.join(target['allowed_tools']) or 'none'}")
PY
STATUS=$?

# Without this the restart note prints even when the edit failed,
# which reads as success.
[ "$STATUS" -eq 0 ] || exit "$STATUS"

echo
echo "  Restart the server to pick this up — principals are read at startup."
