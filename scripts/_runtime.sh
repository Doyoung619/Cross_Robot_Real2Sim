#!/usr/bin/env bash
# Shared project-local runtime paths. Source from a script, never a user shell.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
valid_id() { [[ "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}$ ]]; }
resolve_run() {
  local requested="${1:-}"
  if [[ -z "$requested" ]]; then
    [[ -f "$ROOT/runs/latest_run" ]] || { echo 'No tracked run exists.' >&2; return 2; }
    requested="$(cat "$ROOT/runs/latest_run")"
  fi
  valid_id "$requested" || { echo 'Invalid run ID.' >&2; return 2; }
  RUN_ID="$requested"
  RUN_DIR="$ROOT/runs/$RUN_ID"
  SESSION="crossrobot_$RUN_ID"
  export RUN_ID RUN_DIR SESSION
}
