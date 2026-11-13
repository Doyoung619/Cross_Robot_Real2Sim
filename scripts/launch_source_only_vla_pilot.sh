#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${1:-vla_minimal_$(date -u +%Y%m%dT%H%M%SZ)}"
minutes="${2:-60}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || [[ ! "$minutes" =~ ^[0-9]+$ ]] || ((minutes < 1 || minutes > 60)); then
  echo 'Usage: launch_source_only_vla_pilot.sh [safe_run_id] [minutes: 1..60]' >&2
  exit 2
fi
run_dir="$repo_dir/runs/$run_id"
if [[ -e "$run_dir" ]]; then
  echo "Run already exists; choose a new ID: $run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir"
deadline="$("$repo_dir/.venv/bin/python" -c 'from datetime import datetime,timedelta,timezone;import sys;print((datetime.now(timezone.utc)+timedelta(minutes=int(sys.argv[1]))).isoformat())' "$minutes")"
touch "$run_dir/READY"
printf -v worker_cmd '%q -u %q --run-dir %q --deadline %q > %q 2>&1' \
  "$repo_dir/.venv/bin/python" "$repo_dir/scripts/run_vla_pilot_supervisor.py" "$run_dir" "$deadline" "$run_dir/supervisor.log"
tmux new-session -d -s "crossrobot-$run_id" "$worker_cmd"
ln -sfn "$run_id" "$repo_dir/runs/latest_vla_pilot"
echo "Detached session: crossrobot-$run_id"
echo "Status: $run_dir/status.json"
echo "Log: $run_dir/worker.log"
echo "Result: $run_dir/FINAL_REPORT.md"
