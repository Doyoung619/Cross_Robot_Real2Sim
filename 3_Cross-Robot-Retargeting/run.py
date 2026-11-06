#!/usr/bin/env python3
"""Run stage 3: Cross-Robot Retargeting, using the canonical package implementation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = 3


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 3 — Cross-Robot Retargeting; delegates to the canonical pipeline CLI.")
    parser.add_argument("--config", default="configs/fast.yaml", help="YAML path, relative to the repository root or absolute")
    parser.add_argument("--run-id", help="Existing run ID; stage 1 may start a new run")
    parser.add_argument("--resume", action="store_true", help="Reuse this stage only if its contracts and artifact hashes validate")
    args = parser.parse_args(argv)
    config = Path(args.config).expanduser()
    if not config.is_absolute():
        config = ROOT / config
    command = [str(Path(__file__).resolve()), "stage", "--step", str(STAGE), "--config", str(config)]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.resume:
        command.append("--resume")
    sys.path.insert(0, str(ROOT / "src"))
    from cross_robot_real2sim.cli import main as canonical_main
    previous = sys.argv
    try:
        sys.argv = command
        return canonical_main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
