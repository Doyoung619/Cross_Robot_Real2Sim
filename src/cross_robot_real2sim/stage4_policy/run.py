"""Stage 4 independently runnable entry point."""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import json
import subprocess
import sys
from pathlib import Path
import yaml
from .build_dataset import build_dataset
from .train import train


def run(config: dict, run_dir: Path, *, extra_dirs=()) -> dict:
    run_dir = Path(run_dir)
    config = dict(config)
    stats_path = run_dir / "03_retarget" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        config["policy_tracker_parameters"] = stats.get("policy_tracker_parameters")
    output = run_dir / "04_policy"
    output.mkdir(parents=True, exist_ok=True)
    # A fresh pass must never consume stale model/evaluation/feedback files.
    # Preserve them as historical evidence, outside the current artifact paths.
    previous = [path for path in output.iterdir() if path.name not in ("history", "status.json")]
    previous += [path for path in (run_dir/"videos").glob("target_bc.mp4")]
    if previous and not config.get("resume_training", False):
        archive = output / "history" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive.mkdir(parents=True)
        for path in previous:
            shutil.move(str(path), str(archive / path.name))
    dataset = build_dataset(config, run_dir, extra_dirs)
    if dataset["status"] != "completed":
        result = {"status": "completed_degraded", "training_status": dataset["status"],
                  "reason": "No successfully executed UR5e teacher episode was available.",
                  "dataset": dataset, "privileged_state_used": True, "model_name": "MLP_BC_2x128"}
    else:
        training = train(config, output, resume=bool(config.get("resume_training", False)))
        resolved = output / "effective_config.yaml"
        resolved.write_text(yaml.safe_dump(config))
        videos = run_dir / "videos"
        videos.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "cross_robot_real2sim.stage4_policy.evaluate", "--config", str(resolved),
                   "--output-dir", str(output), "--checkpoint", str(output / "checkpoint.pt"),
                   "--video-dir", str(videos)]
        subprocess.run(command, check=True, timeout=int(config.get("evaluation_timeout_seconds", 1800)))
        evaluation = json.loads((output / "eval_metrics.json").read_text())
        result = {"status": "completed", "dataset": dataset, "training": training, "evaluation": evaluation,
                  "privileged_state_used": True, "model_name": "MLP_BC_2x128"}
    result.update({"schema_version": "1.0", "seed": config.get("seed"), "mode": config.get("mode"),
                   "policy_tracker_parameters": config.get("policy_tracker_parameters"),
                   "input_ids": [str(run_dir / "03_retarget" / "stats.json")]})
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    (output / "status.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(yaml.safe_load(Path(args.config).read_text()), Path(args.run_dir)), indent=2))
