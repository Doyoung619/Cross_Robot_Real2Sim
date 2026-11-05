"""Persist process exit independently of pipeline exceptions or tee failures."""
import datetime
import json
from pathlib import Path
import sys

run_dir, code, interrupted = Path(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))
now = datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def write(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)

stages = {p.parent.name: read(p) for p in sorted(run_dir.glob('0[1-5]_*/status.json'))}
status = read(run_dir / 'status.json')
status.update(exit_code=code, supervisor_exit_code=code, finished_at=now, stage_outcomes=stages)
if interrupted:
    status.update(status='interrupted', reason='tracked_process_received_signal')
elif code:
    status.update(status='failed', reason='runtime_budget_exceeded' if code == 124 else f'pipeline_process_exit_{code}')
elif status.get('status') in (None, 'running', 'launching'):
    status['status'] = 'completed' if len(stages) == 5 and all(v.get('status') == 'completed' for v in stages.values()) else 'partial'
write(run_dir / 'status.json', status)
runtime = read(run_dir / 'runtime.json')
runtime.update(launcher_state='stopped', finished_at=now, exit_code=code)
write(run_dir / 'runtime.json', runtime)
print(f'Run finished: status={status["status"]}, exit_code={code}', flush=True)
