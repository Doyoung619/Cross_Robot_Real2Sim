#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_runtime.sh"
resolve_run "${1:-}"
"$ROOT/.venv/bin/python" - "$RUN_DIR" <<'PY'
import json, os, pathlib, signal, sys
p=pathlib.Path(sys.argv[1]); runtime=json.loads((p/'runtime.json').read_text())
pid=runtime.get('supervisor_pid')
if not pid:
    raise SystemExit('No tracked supervisor PID; no process was signaled.')
try: cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
except FileNotFoundError: raise SystemExit('Tracked run is already stopped.')
if 'run_foreground.sh' not in cmd or p.name not in cmd:
    raise SystemExit('PID identity does not match this project run; no process was signaled.')
os.kill(pid, signal.SIGTERM)
print('Sent SIGTERM to tracked supervisor',pid,'for',p.name)
PY
