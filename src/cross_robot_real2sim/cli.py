from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Traceable five-stage cross-robot simulation MVP")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--config", default="configs/bootstrap.yaml")
    for name in ("run", "stage"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/bootstrap.yaml")
        command.add_argument("--run-id")
        command.add_argument("--resume", action="store_true")
        if name == "stage":
            command.add_argument("--step", type=int, choices=range(1,6), required=True)
    command = commands.add_parser("report")
    command.add_argument("--run-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    os.chdir(root)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.command == "check":
        return subprocess.call([sys.executable, str(root / "scripts" / "preflight.py"), "--config", args.config])
    if args.command == "report":
        from .orchestrator.run_pipeline import report
        print(report(root / "runs" / args.run_id))
        return 0
    from .common.config import load_config
    from .orchestrator.run_pipeline import run_pipeline
    config = load_config(args.config)
    run_id = args.run_id
    if (args.resume or args.command == "stage") and not run_id:
        candidates = [root / "runs" / f"latest_{config.get('mode', 'bootstrap')}"]
        latest_text = root / "runs" / "latest_run"
        if latest_text.exists():
            candidate_id = latest_text.read_text().strip()
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", candidate_id):
                candidates.append(root / "runs" / candidate_id)
        candidates.append(root / "runs" / "latest")
        for candidate in candidates:
            manifest = candidate.resolve() / "manifest.json"
            try:
                previous = json.loads(manifest.read_text())
            except (OSError, ValueError):
                continue
            if previous.get("mode") == config.get("mode", "bootstrap"):
                run_id = candidate.resolve().name
                break
        if not run_id and (args.resume or getattr(args,"step",1) > 1):
            parser.error("No previous run for this mode; provide --run-id for an existing run.")
    run_id = run_id or f"{config.get('mode', 'bootstrap')}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        parser.error("run-id must start with a letter/digit and contain at most 96 letters, digits, underscores or hyphens")
    if args.resume and not (root / "runs" / run_id / "manifest.json").is_file():
        parser.error("Cannot resume a run without a valid manifest; use run without --resume for a new run.")
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    result = run_pipeline(config, root / "runs" / run_id, resume=args.resume, only_step=getattr(args, "step", None))
    print(json.dumps({"run_id": run_id, "status": result["status"], "exit_code": result["exit_code"]}))
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
