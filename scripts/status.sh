#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_runtime.sh"
resolve_run "${1:-}"
"$ROOT/.venv/bin/python" - "$RUN_DIR" <<'PY'
import json, pathlib, os, subprocess, sys
p=pathlib.Path(sys.argv[1]); runtime={}; status={}
for name, target in [('runtime.json',runtime),('status.json',status)]:
    try: target.update(json.loads((p/name).read_text()))
    except (OSError,ValueError): pass
session=runtime.get('session','crossrobot_'+p.name)
tmux=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
pid=runtime.get('supervisor_pid')
alive=False
if pid:
    try:
        cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        alive='run_foreground.sh' in cmd and p.name in cmd
    except OSError: pass
print('Run:',p.name,'session:',session,'tmux_alive:',tmux,'tracked_pid_alive:',alive)
summary={key:status.get(key) for key in ('status','exit_code','started_at','finished_at','current_stage','reason') if key in status}
print('Persisted status:',json.dumps(summary,indent=2))
if not alive and not tmux and 'exit_code' not in status:
    print('STOPPED WITHOUT FINAL EXIT STATUS: inspect logs; success is unverified.')
for s in sorted(p.glob('0[1-5]_*/status.json')):
    try:
        d=json.loads(s.read_text()); result=d.get('result',{}); print(s.parent.name,d.get('status'),'seconds=',round(d.get('duration_seconds',0),2),d.get('reason',result.get('reason',result.get('training_status',''))))
    except (OSError,ValueError): print(s.parent.name,'INVALID STATUS')
PY
if [[ -f "$RUN_DIR/logs/pipeline.log" ]]; then tail -n 30 "$RUN_DIR/logs/pipeline.log"; fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
