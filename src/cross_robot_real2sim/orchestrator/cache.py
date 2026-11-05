"""A file's existence alone is never sufficient for resuming a stage."""
from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def artifact_paths(directory: Path, run_dir: Path) -> set[Path]:
    paths = {path for path in directory.rglob("*")
             if path.is_file() and path.name != "status.json" and not path.name.endswith(".tmp")}
    patterns = {"03_retarget": ("source.mp4", "target_teacher*.mp4"),
                "04_policy": ("target_bc.mp4",), "05_diagnosis": ("comparison.mp4",)}
    for pattern in patterns.get(directory.name, ()):
        paths.update((run_dir / "videos").glob(pattern))
    return paths


def artifact_index(directory: Path, run_dir: Path) -> dict:
    result = {}
    for path in sorted(artifact_paths(directory, run_dir)):
        result[str(path.relative_to(run_dir))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def valid_stage(status_path: Path, run_dir: Path, expected_fingerprint: str) -> bool:
    try:
        status = json.loads(status_path.read_text())
        if status.get("status") not in ("completed", "completed_degraded"):
            return False
        if status.get("input_fingerprint") != expected_fingerprint or not status.get("artifacts"):
            return False
        # Additional files may change a dataset even when all old hashes still
        # match. Require the complete same stage output set on resume.
        current_names = {str(path.relative_to(run_dir)) for path in artifact_paths(status_path.parent, run_dir)}
        if current_names != set(status["artifacts"]):
            return False
        for name, properties in status["artifacts"].items():
            path = run_dir / name
            if not path.is_file() or path.stat().st_size != properties["bytes"] or sha256(path) != properties["sha256"]:
                return False
            if "history" in Path(name).parts:
                # Invalid former outputs remain auditable evidence, but do not
                # have to satisfy the current stage's structural contract.
                continue
            if path.suffix == ".json":
                json.loads(path.read_text())
            elif path.suffix == ".npz":
                import numpy as np
                with np.load(path, allow_pickle=False) as data:
                    for key in data.files:
                        if data[key].dtype == object:
                            return False
            elif path.suffix == ".pt":
                import torch
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if not {"model_state_dict", "model_config", "normalization"} <= checkpoint.keys():
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, EOFError, RuntimeError, pickle.UnpicklingError, zipfile.BadZipFile):
        return False
