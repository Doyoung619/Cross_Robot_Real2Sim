#!/usr/bin/env bash
# Start the measured feedback loop independently of this SSH session.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parent_id="${1:-vla_minimal_20261113T185314Z}"
run_id="${2:-vla_feedback_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$parent_id" =~ ^[A-Za-z0-9_-]+$ ]] || [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'Run IDs must contain only letters, digits, underscores or hyphens.' >&2
  exit 2
fi
parent_dir="$repo_dir/runs/$parent_id"
run_dir="$repo_dir/runs/$run_id"
if [[ -e "$run_dir" || -L "$run_dir" ]]; then
  echo "Run already exists; choose a new ID: $run_dir" >&2
  exit 2
fi
"$repo_dir/.venv/bin/python" - "$repo_dir" "$parent_dir" "$run_dir" <<'PY'
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import sys

repo, parent, out = map(Path, sys.argv[1:])
meta = json.loads((parent / 'training.json').read_text())
if meta.get('steps') != 100:
    raise ValueError('The feedback experiment must start from the original 100-update adapter.')
for name in ('adapter.safetensors', 'processed_source_data.npz', 'selection.json', 'camera_calibration.npz'):
    if not (parent / name).is_file():
        raise FileNotFoundError(parent / name)
adapter_hash = hashlib.sha256((parent / 'adapter.safetensors').read_bytes()).hexdigest()
if adapter_hash != meta.get('adapter_sha256'):
    raise ValueError('Parent adapter checksum does not match its training record.')
started = datetime.now(timezone.utc)
deadline = (started + timedelta(minutes=24)).isoformat()
out.mkdir(parents=True, exist_ok=False)
request = {
    'parent_run': str(parent.relative_to(repo)),
    'rounds': 3,
    'updates_per_round': 100,
    'development_source_ids': ['demo_40', 'demo_41'],
    'final_source_ids': ['demo_47', 'demo_48'],
    'allowed_feedback': 'B development rollout diagnostics guide A selection and offline optimization',
    'training_supervision': 'source-derived only; no B rollout action labels required',
    'requested_at': started.isoformat(),
    'deadline': deadline,
}
(out / 'request.json').write_text(json.dumps(request, indent=2))
(out / 'deadline.txt').write_text(deadline)
(out / 'READY').touch()
PY
deadline="$(cat "$run_dir/deadline.txt")"
printf -v command '%q -u %q --run-dir %q --deadline %q --worker-script scripts/run_vla_feedback.py > %q 2>&1' \
  "$repo_dir/.venv/bin/python" "$repo_dir/scripts/run_vla_pilot_supervisor.py" \
  "$run_dir" "$deadline" "$run_dir/supervisor.log"
tmux new-session -d -c "$repo_dir" -s "crossrobot-$run_id" "$command"
ln -sfn "$run_id" "$repo_dir/runs/latest_vla_feedback"
echo "Detached session: crossrobot-$run_id"
echo "Status: $run_dir/status.json"
echo "Log: $run_dir/worker.log"
echo "Result: $repo_dir/results/$run_id/FINAL_REPORT.md"
