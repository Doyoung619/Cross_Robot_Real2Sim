"""Build a traceable dataset; never accept failed or held-out demonstrations."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def episode_metadata(episode) -> dict:
    for key in ("metadata_json", "metadata"):
        if key in episode:
            value = episode[key]
            return json.loads(str(value.item() if hasattr(value, "item") else value))
    return {}


def evaluation_seeds(config: dict) -> list[int]:
    explicit = config.get("eval_seeds")
    if explicit is not None:
        return [int(seed) for seed in explicit]
    base = int(config.get("eval_seed_base", 10000))
    return list(range(base, base + int(config.get("held_out_eval_seeds", 3))))


def build_dataset(config: dict, run_dir: Path, extra_dirs=()) -> dict:
    roots = [run_dir / "03_retarget" / "rollouts", *map(Path, extra_dirs)]
    held_out = set(evaluation_seeds(config))
    features, actions, episode_indices, provenance = [], [], [], []
    rejected = []
    for path in sorted({p for root in roots for p in root.glob("*.npz")}):
        with np.load(path, allow_pickle=False) as episode:
            meta = episode_metadata(episode)
            if not meta.get("success") or meta.get("robot") != "UR5e":
                continue
            if meta.get("teacher_origin") not in ("source_retargeted", "target_scripted_bootstrap"):
                continue
            seed = int(meta["seed"])
            if seed in held_out:
                raise ValueError(f"Training/evaluation seed overlap: {seed} in {path}")
            x, y = np.asarray(episode["features"], np.float32), np.asarray(episode["actions"], np.float32)
            if x.ndim != 2 or y.shape != (len(x), 4) or not len(x):
                raise ValueError(f"Invalid trajectory dimensions in {path}: {x.shape}, {y.shape}")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite executed data in {path}")
            if features and x.shape[1] != features[0].shape[1]:
                raise ValueError("Inconsistent feature dimensions across demonstrations")
            if np.max(np.abs(y)) > 1.0001:
                raise ValueError("Teacher actions must be normalized into [-1,1]")
            episode_indices.append(np.full(len(x), len(provenance), np.int32))
            features.append(x)
            actions.append(y)
            provenance.append({**meta, "path": str(path), "samples": len(x)})
    if not features:
        return {"status": "blocked_insufficient_teacher_data", "episodes": [], "samples": 0, "rejected": rejected}
    x, y = np.concatenate(features), np.concatenate(actions)
    out = run_dir / "04_policy"
    out.mkdir(parents=True, exist_ok=True)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-3)
    metadata = {
        "schema_version": "1.0", "seed": int(config.get("seed", 42)),
        "mode": config.get("mode", "bootstrap"), "privileged_state_used": True,
        "observation_mode": "oracle_sim_state", "feature_dim": x.shape[1],
        "action_dim": 4, "action_units": "normalized OSC xyz delta; gripper -1=open,+1=close",
        "normalization_source": "training_demonstrations_only", "episodes": provenance,
        "eval_seeds": sorted(held_out), "samples": len(x), "rejected": rejected,
    }
    np.savez_compressed(out / "train_data.npz", features=x, actions=y,
                        episode_indices=np.concatenate(episode_indices), mean=mean, std=std,
                        metadata_json=json.dumps(metadata))
    (out / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"status": "completed", **metadata, "path": str(out / "train_data.npz")}
