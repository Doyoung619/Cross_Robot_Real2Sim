from __future__ import annotations

import json
import shutil
from pathlib import Path
import numpy as np


def corrective_iteration(config: dict, run_dir: Path, rows: list[dict]) -> dict:
    from cross_robot_real2sim.stage3_retarget.rollouts import collect_episode, save_episode
    from cross_robot_real2sim.stage3_retarget.source_trace import load_source_trace
    from cross_robot_real2sim.stage4_policy.run import run as run_policy
    from cross_robot_real2sim.stage4_policy.build_dataset import evaluation_seeds
    output = run_dir / "05_diagnosis"
    before_path = run_dir / "04_policy" / "eval_metrics.json"
    before = json.loads(before_path.read_text()) if before_path.exists() else None
    if before is None:
        return {"status": "intervention_not_executed", "reason": "No learned-policy baseline evaluation exists.",
                "before": None, "after": None, "improvement_claimed": False}
    failures = [row for row in rows if row["teacher_origin"] == "learned_policy" and not row["success"]]
    if not failures:
        return {"status": "intervention_not_executed", "reason": "No observed held-out learned-policy failure to target.",
                "before": before, "after": None, "improvement_claimed": False}
    successful = [row for row in rows if row["success"] and row["teacher_origin"] in ("source_retargeted", "target_scripted_bootstrap")]
    if not successful:
        return {"status": "intervention_not_executed", "reason": "No feasible successful target teacher.",
                "before": before, "after": None, "improvement_claimed": False}
    selection = json.loads((run_dir / "01_selection" / "selection.json").read_text())
    origin = "source_retargeted" if any(row["teacher_origin"] == "source_retargeted" for row in successful) else "target_scripted_bootstrap"
    trace = load_source_trace(selection["source_dataset_path"], selection["selected_source_demonstration_ids"][0],
                              control_hz=config.get("control_hz",20)) if origin == "source_retargeted" else None
    feedback_root = output / "feedback_run"
    rollouts = feedback_root / "03_retarget" / "rollouts"
    rollouts.mkdir(parents=True, exist_ok=True)
    cap = min(int(config.get("feedback_attempt_cap",4)), 6)
    goal = min(int(config.get("feedback_success_goal",2)), cap)
    collected = []
    # Fresh training starts are kept separate from every original held-out seed.
    train_base = int(config.get("seed",42)) + 500
    held_out = set(evaluation_seeds(config))
    for attempt in range(cap):
        seed = train_base + attempt
        if seed in held_out:
            raise ValueError("Feedback collection seed overlaps evaluation seeds")
        result = collect_episode(config, seed=seed, teacher_origin=origin, source_trace=trace)
        meta = save_episode(result, rollouts / f"feedback_{seed}.npz")
        collected.append(meta)
        print(f"Feedback teacher seed={seed}: success={meta.get('success')}", flush=True)
        if sum(bool(item["success"]) for item in collected) >= goal:
            break
    (output / "additional_teacher.json").write_text(json.dumps(collected,indent=2))
    if not any(item["success"] for item in collected):
        return {"status": "intervention_executed_no_new_successes", "intervention": "additional_teacher_collection",
                "before": before, "after": None, "teacher_attempts": len(collected), "teacher_successes": 0,
                "improvement_claimed": False}
    corrected = dict(config)
    corrected["bc_train_steps"] = min(int(config.get("feedback_train_steps",config.get("bc_train_steps",300))),2000)
    corrected["resume_training"] = False
    # Corrective data are merged with the original successfully executed teachers.
    shutil.copytree(run_dir / "03_retarget", feedback_root / "03_retarget", dirs_exist_ok=True)
    learning = run_policy(corrected, feedback_root)
    after = learning.get("evaluation")
    if after is None or after["seeds"] != before["seeds"]:
        raise ValueError("Corrective evaluation must use the identical held-out seeds")
    before_error, after_error = before.get("mean_placement_error"), after.get("mean_placement_error")
    return {"status": "completed", "intervention": "bounded_additional_teacher_collection_and_BC_retraining",
            "seed": config.get("seed"), "schema_version": "1.0", "mode": config.get("mode"),
            "privileged_state_used": True, "teacher_origin": origin,
            "teacher_attempts": len(collected), "teacher_successes": sum(bool(item["success"]) for item in collected),
            "before": before, "after": after, "same_held_out_seeds": before["seeds"],
            "success_rate_delta": after["success_rate"]-before["success_rate"],
            "placement_error_delta_m": after_error-before_error if before_error is not None and after_error is not None else None,
            "improvement_claimed": after["success_rate"] > before["success_rate"]}
