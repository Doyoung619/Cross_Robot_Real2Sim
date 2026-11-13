#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parent_id="${1:-vla_minimal_20261113T185314Z}"
run_id="${2:-vla_extra10m_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$parent_id" =~ ^[A-Za-z0-9_-]+$ ]] || [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'Run IDs must contain only letters, digits, underscores or hyphens.' >&2
  exit 2
fi
parent_dir="$repo_dir/runs/$parent_id"
run_dir="$repo_dir/runs/$run_id"
if [[ -e "$run_dir" ]]; then
  echo "Run already exists: $run_dir" >&2
  exit 2
fi
"$repo_dir/.venv/bin/python" - "$parent_dir" "$run_dir" <<'PY'
from datetime import datetime,timezone,timedelta
from pathlib import Path
import json,sys
parent,out=map(Path,sys.argv[1:])
meta=json.loads((parent/'training.json').read_text())
assert meta['steps']==100,'This comparison expects the original 100-update pilot.'
assert (parent/'processed_source_data.npz').is_file()
out.mkdir(parents=True)
(out/'request.json').write_text(json.dumps({'parent_run':str(parent),'additional_training_seconds':600,'requested_at':datetime.now(timezone.utc).isoformat(),'executed_B_training_demos_allowed':False},indent=2))
(out/'deadline.txt').write_text((datetime.now(timezone.utc)+timedelta(minutes=20)).isoformat())
(out/'READY').touch()
PY
deadline="$(cat "$run_dir/deadline.txt")"
printf -v command '%q -u %q --run-dir %q --deadline %q --worker-script scripts/run_vla_extra_training.py > %q 2>&1; if test -f %q; then %q %q %q > %q 2>&1; fi' \
  "$repo_dir/.venv/bin/python" "$repo_dir/scripts/run_vla_pilot_supervisor.py" "$run_dir" "$deadline" "$run_dir/supervisor.log" \
  "$run_dir/comparison.json" "$repo_dir/.venv/bin/python" "$repo_dir/scripts/report_vla_training_extension.py" "$run_dir" "$run_dir/report.log"
tmux new-session -d -s "crossrobot-$run_id" "$command"
ln -sfn "$run_id" "$repo_dir/runs/latest_vla_training_extension"
echo "Detached run: $run_dir"
echo "Result: $repo_dir/results/$run_id/FINAL_REPORT.md"
