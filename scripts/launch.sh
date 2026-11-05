#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_runtime.sh"
PROFILE="${1:-bootstrap}"
[[ "$PROFILE" == bootstrap || "$PROFILE" == fast || "$PROFILE" == full ]] || { echo "Usage: $0 [bootstrap|fast|full] [run_id] [--resume]" >&2; exit 2; }
resolve_run "${2:-${PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME="${3:-}"
[[ -z "$RESUME" || "$RESUME" == --resume ]] || { echo 'Third argument must be --resume.' >&2; exit 2; }
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Run bash scripts/setup.sh first.' >&2; exit 2; }
command -v tmux >/dev/null || { echo 'tmux is required; install it before launching.' >&2; exit 2; }
mkdir -p "$ROOT/runs" "$RUN_DIR/logs"
exec 9>"$ROOT/runs/.launch.lock"
flock -n 9 || { echo 'Another launch is in progress.' >&2; exit 3; }
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Run already active: $SESSION" >&2; exit 3; fi
if ! flock -n "$ROOT/runs/.gpu0.lock" true; then echo 'A project run already holds GPU 0. Use scripts/status.sh.' >&2; exit 3; fi
if [[ -f "$RUN_DIR/runtime.json" && -z "$RESUME" ]]; then echo 'Existing run ID: use scripts/resume.sh to validate and resume.' >&2; exit 3; fi
# Also guard the small interval between creating a tmux session and taking its lock.
for sid in $(tmux list-sessions -F '#S' 2>/dev/null || true); do
  if [[ "$sid" == crossrobot_* && "$sid" != crossrobot_setup ]]; then
    echo "Another project session exists: $sid. Inspect it before starting GPU work." >&2; exit 3
  fi
done
CONFIG="$ROOT/configs/$PROFILE.yaml"
"$ROOT/.venv/bin/python" - "$RUN_DIR" "$RUN_ID" "$PROFILE" "$SESSION" "$CONFIG" <<'PY'
import json, pathlib, sys, datetime
path, run_id, profile, session, config = sys.argv[1:]
p = pathlib.Path(path)
obj = dict(run_id=run_id, profile=profile, session=session, config=config, launcher_state='launching', launched_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(p/'runtime.json').write_text(json.dumps(obj, indent=2)+'\n')
PY
printf '%s\n' "$RUN_ID" > "$ROOT/runs/latest_run"
ln -sfnT "$RUN_ID" "$ROOT/runs/latest_$PROFILE"
printf -v command_line 'exec env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=%q PYOPENGL_PLATFORM=%q bash %q %q %q %q' "${MUJOCO_GL:-egl}" "${PYOPENGL_PLATFORM:-${MUJOCO_GL:-egl}}" "$ROOT/scripts/run_foreground.sh" "$CONFIG" "$RUN_ID" "$RESUME"
tmux new-session -d -s "$SESSION" "$command_line"
printf 'Run ID: %s\nSession: %s\nLog: %s/logs/pipeline.log\nStatus: bash %q/scripts/status.sh %s\n' "$RUN_ID" "$SESSION" "$RUN_DIR" "$ROOT" "$RUN_ID"
