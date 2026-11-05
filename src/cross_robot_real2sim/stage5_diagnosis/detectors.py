from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from cross_robot_real2sim.stage4_policy.build_dataset import episode_metadata


def diagnose_episode(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as episode:
        meta = episode_metadata(episode)
        positions = np.asarray(episode.get("object_positions", np.empty((0,3))))
        actions = np.asarray(episode.get("actions", np.empty((0,4))))
        contacts = np.asarray(episode.get("grasp_contacts", np.empty(0)))
    lifted = float(meta.get("max_lift_m", np.ptp(positions[:,2]) if len(positions) else 0))
    success = bool(meta.get("native_task_success", meta.get("success", False)))
    drop = bool(not success and len(positions) and lifted >= .06 and
                positions[-1,2] < positions[0,2] + .03)
    release_far = False
    goal = np.asarray(meta.get("goal_position", [np.nan]*3))
    valid_actions = actions.ndim == 2 and actions.shape[1] == 4
    if valid_actions and len(actions) > 1 and len(positions) == len(actions):
        released = np.flatnonzero((actions[:-1,3] > 0) & (actions[1:,3] < 0)) + 1
        release_far = any(positions[t,2] > positions[0,2]+.04 and
                          np.linalg.norm(positions[t,:2]-goal[:2]) > .10 for t in released)
    reason = str(meta.get("reason_code", "unknown"))
    if success:
        reason = "success"
    elif not valid_actions or (len(actions) and (np.max(np.abs(actions)) > 1.001 or not np.isfinite(actions).all())):
        reason = "gripper_action_mismatch"
    elif not len(actions) or not len(positions) or not len(contacts):
        reason = "unknown"
    elif release_far:
        reason = "premature_release"
    elif drop:
        reason = "object_drop"
    elif lifted < .06 and not np.any(contacts) and reason in ("unknown", "grasp_failed"):
        reason = "grasp_failed"
    elif reason == "success":
        reason = "unknown"
    origin = meta.get("teacher_origin", "unknown")
    return {"episode_id": meta.get("episode_id", path.stem), "seed": meta.get("seed"),
            "teacher_origin": origin, "step_attribution": 4 if origin == "learned_policy" else 3,
            "success": success, "reason_code": reason,
            "placement_error_m": meta.get("placement_error_m", meta.get("placement_error")),
            "lift_achieved": lifted >= .06, "max_lift_m": lifted, "object_drop": drop,
            "premature_release": release_far, "unreachable_waypoint": reason == "unreachable_waypoint",
            "collision": meta.get("collision", "unknown"),
            "gripper_action_mismatch": reason == "gripper_action_mismatch",
            "horizon_timeout": reason == "horizon_timeout", "goal_parsing_mismatch": "unknown",
            "source_selection_mismatch": "unknown", "diagnostic_oracle_used": True,
            "confidence_type": "native_simulator_goal_predicate" if success else ("deterministic_rule_on_sim_state" if reason != "unknown" else "unknown"),
            "evidence": json.dumps({"native_task_success": meta.get("native_task_success"),
                                     "max_lift_m": lifted, "grasp_contact_observed": bool(np.any(contacts)),
                                     "final_phase": meta.get("final_phase"), "steps": len(actions)}),
            "rollout_path": str(path)}
