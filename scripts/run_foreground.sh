#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_runtime.sh"
CONFIG="${1:?absolute config path required}"
resolve_run "${2:?run ID required}"
RESUME="${3:-}"
cd "$ROOT"
mkdir -p "$RUN_DIR/logs"
interrupted=0
stop_child() {
  interrupted=1
  if [[ -f "$RUN_DIR/pipeline.pid" ]]; then
    child_pid="$(cat "$RUN_DIR/pipeline.pid")"
    if [[ "$child_pid" =~ ^[0-9]+$ ]]; then "$ROOT/.venv/bin/python" "$ROOT/scripts/stop_pipeline.py" "$child_pid" "$RUN_ID" || true; fi
  fi
}
trap stop_child INT TERM HUP
finalize() {
  local code="$1"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/finalize_status.py" "$RUN_DIR" "$code" "$interrupted" | "$ROOT/.venv/bin/python" -u "$ROOT/scripts/timestamp.py" | tee -a "$RUN_DIR/logs/pipeline.log"
}
trap 'finalize "$?"' EXIT
rm -f "$RUN_DIR/pipeline.pid"
exec 8>"$ROOT/runs/.gpu0.lock"
flock -n 8 || { echo 'GPU 0 is already assigned to another project run.' >&2; exit 3; }
source "$ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1 HF_HOME="$ROOT/.cache/huggingface"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
python - "$RUN_DIR" "$$" <<'PY'
import datetime, json, pathlib, sys
p=pathlib.Path(sys.argv[1])/'runtime.json'
d=json.loads(p.read_text()) if p.exists() else {}
d.update(supervisor_pid=int(sys.argv[2]), launcher_state='running', started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
p.write_text(json.dumps(d,indent=2)+'\n')
PY
printf '[%s] Launching run %s; config=%s resume=%s\n' "$(date -u +%FT%TZ)" "$RUN_ID" "$CONFIG" "$RESUME" | tee -a "$RUN_DIR/logs/pipeline.log"
[[ "$interrupted" == 1 ]] && exit 143
max_seconds="$(python - "$CONFIG" <<'PY_BUDGET'
import sys, yaml
with open(sys.argv[1]) as f: c=yaml.safe_load(f)
minutes=float(c.get('max_runtime_minutes',60))
if not 0 < minutes <= 1440: raise SystemExit('Invalid max_runtime_minutes; must be >0 and <=1440.')
print(max(1,int(minutes*60)))
PY_BUDGET
)"
args=(timeout --verbose --signal=TERM --kill-after=30s "${max_seconds}s" python -u -m cross_robot_real2sim.cli run --config "$CONFIG" --run-id "$RUN_ID")
[[ "$RESUME" == --resume ]] && args+=(--resume)
set +e
(
  printf '%s\n' "$BASHPID" > "$RUN_DIR/pipeline.pid"
  exec setsid "${args[@]}"
) 2>&1 | python -u "$ROOT/scripts/timestamp.py" | tee -a "$RUN_DIR/logs/pipeline.log" &
job_pid=$!
wait "$job_pid"
code=$?
# A trap interrupts wait; wait once more so final status follows child teardown.
if [[ "$interrupted" == 1 ]]; then wait "$job_pid"; code=143; fi
set -e
exit "$code"
