#!/usr/bin/env python3
"""Detached deadline supervisor; records a report even on worker failure/timeout."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (root / args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00")).timestamp()
    status = dict(status="waiting_for_worker_ready", supervisor_pid=os.getpid(),
                  started_at=datetime.now(timezone.utc).isoformat(), deadline=args.deadline,
                  training_data_protocol="source_only_no_executed_B_demonstrations")
    def save():
        temp = out / "status.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(out / "status.json")
    save()
    while not (out / "READY").exists() and time.time() < deadline:
        time.sleep(2)
    returncode = None
    if (out / "READY").exists() and time.time() < deadline:
        status["status"] = "running"
        with (out / "worker.log").open("a", buffering=1) as log:
            process = subprocess.Popen([sys.executable, "-u", str(root / "scripts/run_source_only_vla_pilot.py"),
                                        "--run-dir", str(out), "--deadline", args.deadline],
                                       cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status["worker_pid"] = process.pid
            save()
            while process.poll() is None and time.time() < deadline:
                time.sleep(2)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                status["status"] = "time_budget_exhausted"
            else:
                returncode = process.returncode
                status["status"] = "completed" if returncode == 0 else "worker_failed"
    else:
        status["status"] = "worker_not_ready_before_deadline"
    status.update(exit_code=returncode, finished_at=datetime.now(timezone.utc).isoformat())
    save()
    (out / "exit_code").write_text(str(returncode if returncode is not None else 124)+"\n")
    if not (out / "FINAL_REPORT.md").exists():
        stages = (out / "progress.json").read_text() if (out / "progress.json").exists() else "No worker stages completed."
        (out / "FINAL_REPORT.md").write_text(
            "# One-hour source-only VLA pilot\n\n"
            f"Supervisor status: **{status['status']}**. No successful transfer is claimed.\n\n"
            "Completed stage evidence:\n\n```json\n"+stages+"\n```\n\n"
            "Inspect `worker.log` for the concrete failure or time budget boundary.\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
