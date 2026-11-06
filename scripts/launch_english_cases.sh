#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"
[[ "$PHASE" == all || "$PHASE" == replays ]] || { echo 'Usage: bash scripts/launch_english_cases.sh [all|replays]' >&2; exit 2; }
JOB_ID=english_cases_20261106
SESSION="crossrobot_$JOB_ID"
mkdir -p "$ROOT/runs/$JOB_ID/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Already running: $SESSION" >&2; exit 3; fi
printf -v COMMAND 'env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 %q -u %q --phase %q > %q 2>&1' \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/build_english_cases.py" "$PHASE" "$ROOT/runs/$JOB_ID/logs/english_cases.log"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$COMMAND"
printf 'Session: %s\nStatus: %s/runs/%s/status.json\nLog: %s/runs/%s/logs/english_cases.log\n' "$SESSION" "$ROOT" "$JOB_ID" "$ROOT" "$JOB_ID"
