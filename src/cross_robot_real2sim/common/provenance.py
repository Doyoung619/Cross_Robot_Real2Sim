from datetime import datetime, timezone
import hashlib
from pathlib import Path

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def provenance(config, model_name, input_ids=None, privileged_state_used=False):
    return {
        "schema_version": "1.0", "seed": int(config.get("seed", 42)),
        "mode": config.get("mode", "bootstrap"), "model_controller_name": model_name,
        "input_ids": list(input_ids or []), "privileged_state_used": bool(privileged_state_used),
        "observation_mode": "oracle_sim_state" if privileged_state_used else "input_metadata_or_rgb",
        "created_at": utc_now(), "units": {"length": "meters", "time": "seconds", "angle": "radians"},
    }
