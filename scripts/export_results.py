#!/usr/bin/env python3
"""Export compact, measured run evidence for source control (no datasets/models)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def export(run_id: str):
    from cross_robot_real2sim.orchestrator.run_pipeline import report
    run = ROOT / "runs" / run_id
    status = json.loads((run / "status.json").read_text())
    if status.get("exit_code") is None or status.get("status") == "running":
        raise ValueError("Only terminal runs may be exported")
    report(run)
    destination = ROOT / "results" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    paths = ["FINAL_REPORT.md", "manifest.json", "environment.json", "config.yaml",
             "01_selection/selection.json", "01_selection/source_inspection.json", "02_task/task_spec.json",
             "03_retarget/stats.json", "03_retarget/attempts.csv", "04_policy/eval.csv",
             "04_policy/eval_metrics.json", "04_policy/train_metrics.json", "04_policy/train_log.csv", "04_policy/dataset_metadata.json",
             "05_diagnosis/failure_report.csv", "05_diagnosis/feedback_requests.json",
             "05_diagnosis/before_after.json", "05_diagnosis/metrics.json",
             "05_diagnosis/feedback_run/04_policy/eval.csv", "05_diagnosis/feedback_run/04_policy/eval_metrics.json",
             "videos/source.mp4", "videos/target_teacher.mp4", "videos/target_bc.mp4", "videos/comparison.mp4",
             "05_diagnosis/feedback_run/videos/target_bc.mp4"]
    evidence = {}
    for name in paths:
        source = run / name
        if not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        evidence[name] = {"bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    report_path = destination/"FINAL_REPORT.md"
    if report_path.exists():
        value=report_path.read_text().replace("[03_retarget/source_trace.npz](03_retarget/source_trace.npz)",
                                           "`03_retarget/source_trace.npz` (kept in the raw local run)")
        report_path.write_text(value)
        evidence["FINAL_REPORT.md"]={"bytes":report_path.stat().st_size,"sha256":hashlib.sha256(report_path.read_bytes()).hexdigest(),
                                     "note":"Local NPZ link clarified for the compact export"}
    summary = {key: status.get(key) for key in ("run_id", "status", "exit_code", "started_at", "finished_at", "reason")}
    summary.update(schema_version="1.0", stages={key: {field: value.get(field) for field in
                   ("status", "started_at", "finished_at", "duration_seconds")} for key,value in status.get("stages",{}).items()},
                   raw_local_run=str(run), exported_files=evidence,
                   excluded=["source HDF5", "model checkpoints", "trajectory NPZs", "raw logs", "historical intermediate attempts"])
    (destination/"evidence_manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    for run_id in args.run_ids:
        if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_id):
            raise ValueError("Invalid run ID")
        export(run_id)
