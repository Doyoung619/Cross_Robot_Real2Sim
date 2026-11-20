#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-pilot}"
run_id="${2:-benchmark_${profile}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$profile" =~ ^(smoke|pilot|full)$ ]] || [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then echo 'Invalid profile or run ID' >&2; exit 2; fi
run_dir="$repo_dir/runs/$run_id"
if [[ -e "$run_dir" || -L "$run_dir" ]]; then echo 'Run already exists' >&2; exit 2; fi
printf -v command '%q -u -m cross_robot_real2sim.benchmark supervise --config %q --run-dir %q > %q 2>&1' \
 "$repo_dir/.venv/bin/python" "$repo_dir/configs/benchmark/$profile.json" "$run_dir" "$repo_dir/runs/${run_id}_supervisor.log"
if (( $# > 2 )); then
  method_args=("${@:3}")
  for method in "${method_args[@]}"; do if [[ ! "$method" =~ ^[a-z_]+$ ]]; then echo 'Invalid method ID' >&2; exit 2; fi; done
  printf -v suffix ' %q' --methods "${method_args[@]}"
  printf -v command '%q -u -m cross_robot_real2sim.benchmark supervise --config %q --run-dir %q%s > %q 2>&1' \
    "$repo_dir/.venv/bin/python" "$repo_dir/configs/benchmark/$profile.json" "$run_dir" "$suffix" "$repo_dir/runs/${run_id}_supervisor.log"
fi
tmux new-session -d -c "$repo_dir" -s "crossrobot-$run_id" "$command"
echo "Session: crossrobot-$run_id"
echo "Status: $run_dir/status.json"
echo "Log: $run_dir/worker.log"
echo "Results: $repo_dir/results/$run_id/README.md"
