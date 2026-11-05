"""Validate actual integrated artifacts, optionally launching a detached bootstrap.

Set CROSSROBOT_TEST_RUN to the completed run ID to inspect existing evidence.
Without that setting this integration test starts the real bootstrap via tmux.
"""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.sim
def test_real_bootstrap_artifact_integrity_and_provenance():
    run_id = os.environ.get("CROSSROBOT_TEST_RUN")
    if run_id is None:
        run_id = f"bootstrap_test_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
        subprocess.run(["bash", "scripts/launch.sh", "bootstrap", run_id], cwd=ROOT, check=True, capture_output=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            runtime = ROOT / "runs" / run_id / "runtime.json"
            if runtime.exists() and json.loads(runtime.read_text()).get("launcher_state") == "stopped":
                break
            time.sleep(.25)
        else:
            pytest.fail(f"Detached pipeline still running; inspect bash scripts/status.sh {run_id}")
    run_dir = ROOT / "runs" / run_id
    status = json.loads((run_dir / "status.json").read_text())
    assert status["exit_code"] == 0
    assert len(status["stages"]) == 5
    assert all(item["status"] in ("completed", "completed_degraded") for item in status["stages"].values())
    selection = json.loads((run_dir / "01_selection/selection.json").read_text())
    task = json.loads((run_dir / "02_task/task_spec.json").read_text())
    assert Path(selection["source_dataset_path"]).stat().st_size > 1_000_000
    assert task["task_source"] == "provided_seed" and task["goal_pose_source"] == "simulator_oracle"
    stats = json.loads((run_dir / "03_retarget/stats.json").read_text())
    assert stats["counts"]["source_scripted"]["attempts"] >= 1
    assert stats["counts"]["source_retargeted"]["attempts"] >= 1
    assert stats["target_teacher_success_count"] >= 1
    with np.load(run_dir / "04_policy/train_data.npz", allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"].item()))
        assert data["features"].shape[1] == 26 and data["actions"].shape[1] == 4
        assert not {item["seed"] for item in meta["episodes"]} & set(meta["eval_seeds"])
        assert all(item["success"] and item["robot"] == "UR5e" for item in meta["episodes"])
    evaluation = json.loads((run_dir / "04_policy/eval_metrics.json").read_text())
    assert evaluation["fresh_process_evaluation"] and evaluation["episodes"] >= 3
    assert 0 <= evaluation["success_rate"] <= 1
    iteration = json.loads((run_dir / "05_diagnosis/before_after.json").read_text())
    if iteration["status"] == "completed":
        assert iteration["before"]["seeds"] == iteration["after"]["seeds"]
    for name in ("source.mp4", "target_teacher.mp4", "target_bc.mp4", "comparison.mp4"):
        path = run_dir / "videos" / name
        assert path.stat().st_size > 1000
        import imageio.v2 as imageio
        reader = imageio.get_reader(path)
        try:
            assert reader.get_data(0).shape[-1] == 3
        finally:
            reader.close()
    assert (run_dir / "FINAL_REPORT.md").is_file()
