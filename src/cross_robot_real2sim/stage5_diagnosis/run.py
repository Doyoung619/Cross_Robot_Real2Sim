from __future__ import annotations

import argparse
import datetime as dt
import shutil
import csv
import json
import time
import traceback
from pathlib import Path
from .detectors import diagnose_episode
from .feedback import make_feedback
from .retrain_check import corrective_iteration


def run(config: dict, run_dir: Path) -> dict:
    start = time.monotonic()
    run_dir = Path(run_dir)
    output = run_dir / "05_diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    # A fresh pass must never consume stale model/evaluation/feedback files.
    # Preserve them as historical evidence, outside the current artifact paths.
    previous = [path for path in output.iterdir() if path.name not in ("history", "status.json")]
    previous += [path for path in (run_dir/"videos").glob("comparison.mp4")]
    if previous:
        archive = output / "history" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive.mkdir(parents=True)
        for path in previous:
            shutil.move(str(path), str(archive / path.name))
    paths = sorted((run_dir/"03_retarget"/"rollouts").glob("*.npz")) + sorted((run_dir/"04_policy"/"eval_rollouts").glob("*.npz"))
    rows = [diagnose_episode(path) for path in paths]
    feedback = make_feedback(rows, config)
    (output / "feedback_requests.json").write_text(json.dumps(feedback, indent=2))
    try:
        correction = corrective_iteration(config, run_dir, rows)
    except Exception as exc:
        correction = {"status": "intervention_failed", "exception": repr(exc), "traceback": traceback.format_exc(),
                      "improvement_claimed": False}
        print(correction["traceback"], flush=True)
    correction.update(schema_version="1.0", seed=config.get("seed"), mode=config.get("mode"),
                      model_name="deterministic_simulator_diagnostics", diagnostic_oracle_used=True)
    (output / "before_after.json").write_text(json.dumps(correction, indent=2))
    additional = output / "feedback_run"
    for path in sorted((additional/"03_retarget"/"rollouts").glob("feedback_*.npz")) + sorted((additional/"04_policy"/"eval_rollouts").glob("*.npz")):
        row = diagnose_episode(path)
        row["episode_id"] = "feedback_" + row["episode_id"]
        rows.append(row)
    fieldnames = list(rows[0]) if rows else ["episode_id", "reason_code", "evidence"]
    with (output/"failure_report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    # Build visual evidence after the learned-policy video exists.
    comparison_error = None
    try:
        from cross_robot_real2sim.viz.comparison_video import create_comparison
        create_comparison(run_dir)
    except Exception as exc:
        comparison_error = repr(exc)
    result = {"status": "completed" if rows and correction["status"] == "completed" else "completed_degraded",
              "schema_version": "1.0", "seed": config.get("seed"), "mode": config.get("mode"),
              "model_name": "deterministic_simulator_diagnostics", "privileged_state_used": True,
              "diagnostic_oracle_used": True, "input_ids": [str(path) for path in paths],
              "episodes_analyzed": len(rows), "correction_status": correction["status"],
              "comparison_video_error": comparison_error, "duration_seconds": time.monotonic()-start}
    (output/"metrics.json").write_text(json.dumps(result,indent=2))
    (output/"status.json").write_text(json.dumps(result,indent=2))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args=parser.parse_args()
    from cross_robot_real2sim.common.config import load_config
    print(json.dumps(run(load_config(args.config),Path(args.run_dir)),indent=2))
