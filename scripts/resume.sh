#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_runtime.sh"
PROFILE="${1:-bootstrap}"
requested="${2:-}"
if [[ -z "$requested" && -L "$ROOT/runs/latest_$PROFILE" ]]; then
  requested="$(basename -- "$(readlink -f -- "$ROOT/runs/latest_$PROFILE")")"
fi
resolve_run "$requested"
# Stable run ID and persisted configuration; CLI validates hashes/schema/seed split.
if [[ -f "$RUN_DIR/runtime.json" ]]; then
  recorded_profile="$("$ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("profile",sys.argv[2]))' "$RUN_DIR/runtime.json" "$PROFILE")"
  [[ "$recorded_profile" == "$PROFILE" ]] || { echo "Run profile is $recorded_profile, not $PROFILE." >&2; exit 2; }
fi
exec bash "$ROOT/scripts/launch.sh" "$PROFILE" "$RUN_ID" --resume
