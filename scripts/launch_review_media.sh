#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_ID="${1:-review_media_$(date -u +%Y%m%dT%H%M%SZ)}"
PHASE="${2:-all}"
[[ "$JOB_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo 'Invalid job ID' >&2; exit 2; }
[[ "$PHASE" == all || "$PHASE" == replays ]] || { echo 'Phase must be all or replays' >&2; exit 2; }
SESSION="crossrobot_$JOB_ID"
mkdir -p "$ROOT/runs/$JOB_ID/logs"
exec 9>"$ROOT/runs/.launch.lock"
flock -n 9 || { echo 'Another launch is in progress' >&2; exit 3; }
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Already running: $SESSION" >&2; exit 3; fi
flock -n "$ROOT/runs/.gpu0.lock" true || { echo 'Another project job holds GPU 0' >&2; exit 3; }
printf -v COMMAND 'env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 %q -u %q --job-id %q --phase %q > %q 2>&1' \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/build_review_media.py" "$JOB_ID" "$PHASE" "$ROOT/runs/$JOB_ID/logs/review.log"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$COMMAND"
printf 'Job: %s\nSession: %s\nStatus: %s/runs/%s/status.json\nLog: %s/runs/%s/logs/review.log\n' "$JOB_ID" "$SESSION" "$ROOT" "$JOB_ID" "$ROOT" "$JOB_ID"
