from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
import yaml
from .cache import artifact_index, fingerprint, sha256, valid_stage
from .run_state import atomic_json, now, stage_directory, archive_stage_outputs

LOG = logging.getLogger(__name__)
STAGES = {1: "stage1_selection", 2: "stage2_task", 3: "stage3_retarget", 4: "stage4_policy", 5: "stage5_diagnosis"}


def environment() -> dict:
    packages = {}
    for name in ("torch", "numpy", "robosuite", "mujoco", "h5py", "transformers", "pydantic", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.free", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        gpu = "unavailable"
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "gpu": gpu, "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "created_at": now()}


def input_signature(config: dict, root: Path):
    from cross_robot_real2sim.adapters.source_hdf5 import dataset_path, ensure_dataset
    source = Path(dataset_path(config))
    if not source.exists():
        source = Path(ensure_dataset(config))
    from cross_robot_real2sim.common.config import resolve_path
    video_dir = resolve_path(config.get("human_video_dir", "data/human"), root)
    videos = [{"path": str(p), "sha256": sha256(p)} for p in sorted(video_dir.iterdir())
              if p.is_file() and p.suffix.lower() == ".mp4"] if video_dir.exists() else []
    task_seed = resolve_path(config.get("task_seed", "data/task_seeds/can_to_bin.json"), root)
    return {"source_dataset": {"path": str(source), "sha256": sha256(source) if source.exists() else "missing"},
            "human_videos": videos, "task_seed": sha256(task_seed) if task_seed.exists() else "missing"}


def implementation_signature(root: Path, step: int) -> dict:
    package = root / "src" / "cross_robot_real2sim"
    directories = [package / "common", package / "adapters", package / STAGES[step]]
    if step >= 4:
        directories.append(package / "stage3_retarget")
    if step == 5:
        directories.extend([package / "stage4_policy", package / "viz"])
    paths = {package / "schemas.py"}
    for directory in directories:
        paths.update(directory.glob("*.py"))
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths) if path.exists()}


def run_pipeline(config: dict, run_dir: Path, *, resume=False, only_step=None) -> dict:
    run_dir = Path(run_dir).resolve()
    root = Path(__file__).resolve().parents[3]
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = fingerprint(config)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("config_fingerprint") != config_hash:
            raise ValueError("Run config differs from frozen manifest; use a new run ID.")
        if not resume and only_step is None:
            raise ValueError("Run ID already exists. Resume it or use a new ID.")
    else:
        atomic_json(manifest_path, {"schema_version": "1.0", "run_id": run_dir.name,
                    "created_at": now(), "config": config, "config_fingerprint": config_hash,
                    "seed": config.get("seed"), "mode": config.get("mode"),
                    "observation_mode": "oracle_sim_state", "privileged_state_used": True})
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    if not (run_dir / "environment.json").exists():
        atomic_json(run_dir / "environment.json", environment())
    signature = input_signature(config, root)
    status = {"schema_version": "1.0", "run_id": run_dir.name, "status": "running", "started_at": now(),
              "pid": os.getpid(), "stages": {}, "mode": config.get("mode"), "exit_code": None}
    for step in range(1, 6):
        existing = run_dir / stage_directory(step) / "status.json"
        if existing.exists():
            try:
                status["stages"][str(step)] = json.loads(existing.read_text())
            except ValueError:
                pass
    atomic_json(run_dir / "status.json", status)
    for step in ([only_step] if only_step else range(1, 6)):
        used = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if used > float(config.get("max_run_disk_gb", 5)) * 1024**3:
            status.update(status="failed", exit_code=1, finished_at=now(), reason="run_disk_budget_exceeded")
            atomic_json(run_dir / "status.json", status)
            report(run_dir)
            return status
        stage_dir = run_dir / stage_directory(step)
        stage_dir.mkdir(exist_ok=True)
        dependencies = {key: value.get("artifacts", {}) for key, value in status["stages"].items() if int(key) < step}
        expected = fingerprint({"config": config_hash, "inputs": signature, "dependencies": dependencies, "step": step,
                                "implementation": implementation_signature(root, step)})
        stage_status_path = stage_dir / "status.json"
        if resume and valid_stage(stage_status_path, run_dir, expected):
            LOG.info("Stage %s: resuming verified outputs", step)
            continue
        if not (step == 4 and config.get("resume_training",False)):
            archive_stage_outputs(run_dir, step)
        status["current_stage"] = step
        stage_status = {"schema_version": "1.0", "step": step, "status": "running", "started_at": now(),
                        "seed": config.get("seed"), "mode": config.get("mode"), "input_fingerprint": expected}
        status["stages"][str(step)] = stage_status
        atomic_json(stage_status_path, stage_status)
        atomic_json(run_dir / "status.json", status)
        start = time.monotonic()
        LOG.info("Stage %s (%s) started", step, STAGES[step])
        try:
            if step == 2 and status["stages"].get("1",{}).get("status") not in ("completed","completed_degraded"):
                result = {"status": "skipped_missing_input", "reason": "valid_selection_unavailable"}
                atomic_json(stage_dir / "missing_input.json", result)
            elif step in (3, 4) and config.get("mode") == "full" and not signature["human_videos"]:
                result = {"status": "skipped_missing_input", "reason": "requires_human_video"}
                atomic_json(stage_dir / "missing_input.json", result)
            elif step in (3, 4) and status["stages"].get("2", {}).get("status") not in ("completed", "completed_degraded"):
                result = {"status": "skipped_missing_input", "reason": "valid_task_spec_unavailable"}
                atomic_json(stage_dir / "missing_input.json", result)
            else:
                module = importlib.import_module(f"cross_robot_real2sim.{STAGES[step]}.run")
                result = module.run(config, run_dir)
            stage_status.update({"status": result.get("status", "completed"), "result": result,
                                 "finished_at": now(), "duration_seconds": time.monotonic() - start})
            if stage_status["status"] not in ("completed", "completed_degraded", "failed", "skipped_missing_input"):
                stage_status["result_status"] = stage_status["status"]
                stage_status["status"] = "completed_degraded"
        except (Exception, KeyboardInterrupt) as exc:
            LOG.exception("Stage %s failed", step)
            stage_status.update({"status": "failed", "exception": repr(exc), "traceback": traceback.format_exc(),
                                 "finished_at": now(), "duration_seconds": time.monotonic() - start})
            if isinstance(exc, KeyboardInterrupt):
                stage_status["artifacts"] = artifact_index(stage_dir, run_dir)
                atomic_json(stage_status_path, stage_status)
                status.update(status="interrupted", exit_code=130, finished_at=now())
                atomic_json(run_dir / "status.json", status)
                raise
        stage_status["artifacts"] = artifact_index(stage_dir, run_dir)
        atomic_json(stage_status_path, stage_status)
        atomic_json(run_dir / "status.json", status)
        LOG.info("Stage %s: %s in %.1fs", step, stage_status["status"], stage_status["duration_seconds"])
    states = [entry["status"] for entry in status["stages"].values()]
    terminal = "failed" if "failed" in states else ("completed_degraded" if any(s != "completed" for s in states) else "completed")
    if config.get("mode") == "full" and not signature["human_videos"]:
        terminal = "partial"
        status["reason"] = "requires_human_video"
    status.update(status=terminal, exit_code=1 if "failed" in states else 0, finished_at=now())
    atomic_json(run_dir / "status.json", status)
    report(run_dir)
    return status


def report(run_dir: Path) -> Path:
    def read(relative):
        path = run_dir / relative
        return json.loads(path.read_text()) if path.exists() else {}
    manifest = read("manifest.json")
    status = read("status.json")
    stage3 = read("03_retarget/stats.json")
    stage4 = read("04_policy/metrics.json")
    correction = read("05_diagnosis/before_after.json")
    env = read("environment.json")
    lines = [f"# Cross-Robot Real2Sim — {run_dir.name}", "",
             f"Execution: **{status.get('status', 'unknown')}**, exit `{status.get('exit_code')}`; mode `{manifest.get('mode')}`. "
             "State input: **oracle_sim_state**. Privileged simulator object poses/contact are used.", "",
             "## Actual stage outcomes", "", "| Stage | Outcome | Seconds | Evidence |", "|---|---|---:|---|"]
    for step in range(1,6):
        entry = status.get("stages", {}).get(str(step), {})
        result = entry.get("result", {})
        reason = entry.get("exception", result.get("reason", result.get("training_status", "")))
        if step == 3:
            reason = f"{stage3.get('source_trace_count',0)} real source traces, {stage3.get('target_teacher_success_count',0)} successful target demos"
        elif step == 4 and stage4.get("evaluation"):
            metric = stage4["evaluation"]
            reason = f"Learned BC {metric.get('successes')}/{metric.get('episodes')} held-out placements; fresh process"
        elif step == 5:
            reason = correction.get("status", reason)
        lines.append(f"| {step}: {STAGES[step]} | {entry.get('status', 'not_run')} | {entry.get('duration_seconds',0):.2f} | {reason} |")
    counts = stage3.get("counts", {})
    lines += ["", "## Measured placements", "", "| Controller/data origin | Successes / attempts |", "|---|---:|"]
    for key,label in (("source_scripted","Panda scripted source policy"),
                      ("source_retargeted","UR5e genuine source-grounded waypoint retargeting"),
                      ("target_scripted_bootstrap","UR5e independent scripted fallback")):
        count = counts.get(key,{})
        lines.append(f"| {label} | {count.get('successes',0)} / {count.get('attempts',0)} |")
    for label,metric in (("UR5e learned BC, before correction", stage4.get("evaluation",{})),
                         ("UR5e learned BC, after correction", correction.get("after") or {})):
        lines.append(f"| {label} | {metric.get('successes','not run')} / {metric.get('episodes','not run')} |")
    if stage3 and not stage3.get("genuine_transfer_verified"):
        lines += ["", "**End-to-end infrastructure working; genuine cross-robot transfer NOT YET verified.**"]
    elif not stage3:
        lines += ["", f"Retargeting was not evaluated in this run: `{status.get('reason','see stage outcomes')}`."]
    eval_metric = stage4.get("evaluation",{})
    after = correction.get("after") or {}
    lines += ["", f"BC training samples: {stage4.get('dataset',{}).get('samples',0)}; "
              f"feature dimension: {stage4.get('training',{}).get('feature_dim','not trained')}; "
              f"training steps: {stage4.get('training',{}).get('steps',0)}.",
              f"Held-out seeds: `{eval_metric.get('seeds',[])}`. "
              f"Mean final XY placement error: `{eval_metric.get('mean_placement_error')}` m before, "
              f"`{after.get('mean_placement_error')}` m after.",
              f"Correction: `{correction.get('status','not run')}`, "
              f"additional teachers {correction.get('teacher_successes',0)}/{correction.get('teacher_attempts',0)}; "
              f"same held-out seeds: `{correction.get('same_held_out_seeds',[])}`. "
              "These small samples do not establish statistical generalization.", "",
              "## Reproduce", "", "```bash", "bash scripts/setup.sh", "bash scripts/preflight.sh",
              f"bash scripts/launch.sh {manifest.get('mode','bootstrap')}", f"bash scripts/status.sh {run_dir.name}",
              f"bash scripts/resume.sh {manifest.get('mode','bootstrap')} {run_dir.name}", f"tail -f runs/{run_dir.name}/logs/pipeline.log",
              f"bash scripts/stop.sh {run_dir.name}", "```", "", "## Environment and inputs", "",
              f"Python: `{env.get('python','').splitlines()[0]}`. GPU: `{env.get('gpu')}`.",
              f"Packages: `{json.dumps(env.get('packages',{}),sort_keys=True)}`. Exact lock: `requirements-working-bootstrap.lock`.",
              f"Dataset: `{stage3.get('source_dataset_path','data/source/v1.5/can/ph/low_dim_v15.hdf5')}`. "
              f"Selected source IDs: `{stage3.get('selected_source_demonstration_ids',[])}`.", "",
              "- Source: public robomimic Can / PH / v1.5 low_dim_v15.hdf5; source data is human robot teleoperation, not a human video.",
              "- Bootstrap task source: explicitly provided manual task seed. Robot trajectories and placement are executed in MuJoCo.",
              "- Real human video, CLIP, Qwen inference and optional SAM2 are untested unless a full-mode report explicitly records their execution.",
              "- SAM2 is an unimplemented optional extension. No vision-only transfer, physical robot execution, or real-world transfer is claimed.",
              "- Source EEF orientation is preserved in source traces; target OSC holds the initial downward grip orientation, with disclosed grip-site calibration.",
              "- BC uses a shared online event/geometry phase tracker; it is not an end-to-end vision policy. Timed gripper dwell is part of this controller.", "",
              "## Evidence files", ""]
    for rel in ("01_selection/selection.json", "02_task/task_spec.json", "03_retarget/source_trace.npz", "03_retarget/attempts.csv",
                "04_policy/dataset_metadata.json", "04_policy/eval.csv", "05_diagnosis/failure_report.csv",
                "05_diagnosis/feedback_requests.json", "05_diagnosis/before_after.json"):
        if (run_dir/rel).exists():
            lines.append(f"- [{rel}]({rel})")
    videos = sorted((run_dir/"videos").glob("*.mp4"))
    for path in videos:
        lines.append(f"- [{path.name}](videos/{path.name}) ({path.stat().st_size} bytes)")
    # The diagnosis ranks only supported observations, rather than inventing bottlenecks.
    feedback = read("05_diagnosis/feedback_requests.json")
    lines += ["", "## Ranked bottlenecks", ""]
    for item in feedback.get("ranked_bottlenecks", []):
        lines.append(f"- {item['reason_code']}: {item['count']} observed episode(s). {item.get('next_experiment','')}")
    if len(feedback.get("ranked_bottlenecks",[])) < 3:
        lines.append(f"- Limited training coverage: {len(stage4.get('dataset',{}).get('episodes',[]))} successful target teachers "
                     f"from {stage3.get('source_trace_count',0)} selected source demos. Hypothesis: more executed state coverage reduces BC distribution shift; test on new seeds.")
    if len(feedback.get("ranked_bottlenecks",[])) < 2:
        selection = read("01_selection/selection.json")
        lines.append(f"- Human-video branch: `{selection.get('human_clip_selection','not run')}`. "
                     "A real can-to-bin human clip is needed to measure CLIP/Qwen selection and goal-parsing errors.")
    if not videos:
        lines.append("No video evidence was produced; see stage exception logs.")
    path = run_dir / "FINAL_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path
