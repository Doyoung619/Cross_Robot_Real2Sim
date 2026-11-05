from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from pathlib import Path


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False))
    os.replace(temporary, path)


def stage_directory(step: int) -> str:
    return {1: "01_selection", 2: "02_task", 3: "03_retarget", 4: "04_policy", 5: "05_diagnosis"}[step]


def archive_stage_outputs(run_dir: Path, step: int):
    """Keep invalid former evidence away from every current output contract."""
    directory = run_dir / stage_directory(step)
    previous = [path for path in directory.iterdir() if path.name not in ("history", "status.json")]
    patterns = {3: ("source.mp4", "target_teacher*.mp4"), 4: ("target_bc.mp4",), 5: ("comparison.mp4",)}
    external = [path for pattern in patterns.get(step,()) for path in (run_dir/"videos").glob(pattern)]
    if previous or external:
        archive = directory / "history" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive.mkdir(parents=True)
        for path in previous:
            shutil.move(str(path), str(archive/path.name))
        if external:
            (archive/"videos").mkdir()
            for path in external:
                shutil.move(str(path), str(archive/"videos"/path.name))
