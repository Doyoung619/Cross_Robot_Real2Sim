"""Terminate only a CLI process group whose live identity matches this run."""
import os
from pathlib import Path
import signal
import sys
pid, run_id = int(sys.argv[1]), sys.argv[2]
try:
    args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    args = [arg.decode(errors='replace') for arg in args if arg]
except FileNotFoundError:
    raise SystemExit(0)
if 'cross_robot_real2sim.cli' not in args or '--run-id' not in args:
    raise SystemExit('Child PID identity mismatch; no signal sent.')
index = args.index('--run-id')
if index+1 >= len(args) or args[index+1] != run_id:
    raise SystemExit('Child run ID mismatch; no signal sent.')
try:
    if os.getpgid(pid) == pid:
        os.killpg(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)
except ProcessLookupError:
    pass
