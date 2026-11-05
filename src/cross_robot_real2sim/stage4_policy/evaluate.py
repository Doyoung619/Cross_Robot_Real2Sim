"""Fresh-process closed-loop evaluation of a saved learned model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .bc_model import load_policy
from .build_dataset import evaluation_seeds


def evaluate(config: dict, output_dir: Path, checkpoint: Path, *, video_dir: Path | None = None) -> dict:
    from cross_robot_real2sim.stage3_retarget.rollouts import run_policy_episode
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_rollouts").mkdir(exist_ok=True)
    predict, payload = load_policy(checkpoint, device="cpu")
    train_seeds = {int(ep["seed"]) for ep in payload["dataset_metadata"]["episodes"]}
    seeds = evaluation_seeds(config)
    if train_seeds & set(seeds):
        raise ValueError("Held-out seeds overlap demonstration seeds")
    records = []
    for index, seed in enumerate(seeds):
        video = video_dir / "target_bc.mp4" if video_dir and index == 0 else None
        episode = run_policy_episode(config, predict, seed=seed, video_path=video)
        meta = dict(episode.get("metadata", episode.get("metrics", {})))
        if "metadata_json" in episode:
            raw = episode["metadata_json"]
            meta.update(json.loads(str(raw.item() if hasattr(raw, "item") else raw)))
        meta.update({"seed": seed, "teacher_origin": "learned_policy", "robot": "UR5e",
                     "schema_version": "1.0", "mode": config.get("mode"),
                     "privileged_state_used": True, "checkpoint": str(checkpoint)})
        path = output_dir / "eval_rollouts" / f"eval_{seed}.npz"
        arrays = {k: v for k, v in episode.items() if isinstance(v, np.ndarray) and v.dtype != object}
        np.savez_compressed(path, **{k:v for k,v in arrays.items() if k != "metadata_json"}, metadata_json=json.dumps(meta))
        record = {"seed": seed, "success": bool(meta.get("success", False)),
                  "placement_error": meta.get("placement_error_m", meta.get("placement_error", meta.get("distance_to_goal", float("nan")))),
                  "lift_achieved": meta.get("lift_achieved", float(meta.get("max_lift_m", 0)) >= 0.06),
                  "steps": meta.get("steps", len(arrays.get("actions", []))),
                  "failure_code": meta.get("reason_code", meta.get("failure_code", "unknown")),
                  "teacher_origin": "learned_policy", "robot": "UR5e",
                  "rollout_path": str(path), "diagnostic_oracle_used": True}
        records.append(record)
        print(f"BC eval seed={seed}: {record}", flush=True)
    with (output_dir / "eval.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else ["seed", "success"])
        writer.writeheader()
        writer.writerows(records)
    errors = [float(row["placement_error"]) for row in records if np.isfinite(float(row["placement_error"]))]
    metrics = {"status": "completed", "model_name": "MLP_BC_2x128", "seed": config.get("seed"),
               "mode": config.get("mode"), "schema_version": "1.0", "privileged_state_used": True,
               "teacher_origin": "learned_policy", "seeds": seeds, "episodes": len(records),
               "successes": sum(row["success"] for row in records),
               "success_rate": sum(row["success"] for row in records) / len(records) if records else None,
               "mean_placement_error": float(np.mean(errors)) if errors else None,
               "checkpoint": str(checkpoint), "fresh_process_evaluation": True}
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-dir")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    evaluate(config, Path(args.output_dir), Path(args.checkpoint),
             video_dir=Path(args.video_dir) if args.video_dir else None)


if __name__ == "__main__":
    main()
