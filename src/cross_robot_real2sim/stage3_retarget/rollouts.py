from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
from ..adapters.robosuite_env import make_env, observation, render_frame, goal_success, safety_diagnostics
from ..adapters.robot_action_adapter import RobotActionAdapter
from .controller import PhaseTracker, build_features, teacher_action, PHASES, FEATURE_NAMES
from .waypoints import transfer_parameters


def _episode(config, seed, robot, teacher_origin, source_trace=None, policy_callable=None, video_path=None):
    started = time.monotonic()
    env = make_env(config, robot=robot, seed=seed)
    records = {k: [] for k in ("features", "actions", "object_positions", "eef_positions", "timestamps", "phases", "grasp_contacts","arm_collisions","joint_limit_margins_rad")}
    collision_pairs=set()
    frames = []
    render_error = None
    try:
        adapter = RobotActionAdapter(env)
        obs = observation(env)
        params = transfer_parameters(source_trace, teacher_origin,
                                     clearance=float(config.get("grasp_clearance", .18)))
        policy_params = config.get("policy_tracker_parameters") or config.get("policy_transfer_parameters")
        if policy_callable is not None and policy_params:
            params = dict(policy_params)
            params["grasp_offset"] = np.asarray(params["grasp_offset"], dtype=float)
        tracker = PhaseTracker(obs, params)
        initial_object = obs["object_pos"].copy()
        goal = obs["goal_pos"].copy()
        max_lift = 0.
        success = False
        for step in range(int(config.get("horizon", 300))):
            tracker.update(obs)
            features = build_features(obs, env, tracker)
            action = teacher_action(obs, tracker, adapter) if policy_callable is None else np.asarray(policy_callable(features), dtype=float)
            if action.shape != (4,) or not np.all(np.isfinite(action)):
                raise ValueError(f"Invalid policy output {action}")
            action = np.clip(action, -1, 1)
            records["features"].append(features)
            records["actions"].append(action)
            records["object_positions"].append(obs["object_pos"].copy())
            records["eef_positions"].append(obs["eef_pos"].copy())
            records["timestamps"].append(step / float(config.get("control_hz",20)))
            records["phases"].append(tracker.phase)
            records["grasp_contacts"].append(obs["grasp_contact"])
            diagnostics=safety_diagnostics(env)
            records["arm_collisions"].append(diagnostics["arm_collision"])
            records["joint_limit_margins_rad"].append(diagnostics["joint_limit_margin_rad"])
            collision_pairs.update(diagnostics["arm_collision_pairs"])
            if video_path and step % int(config.get("render_stride", 2)) == 0 and render_error is None:
                try:
                    frames.append(render_frame(env, config))
                except Exception as exc:
                    render_error = str(exc)
            env.step(adapter.to_env(action))
            obs = observation(env)
            max_lift = max(max_lift, float(obs["object_pos"][2] - initial_object[2]))
            success = goal_success(env, obs)
            if success and tracker.phase >= 7:
                break
        error = float(np.linalg.norm(obs["object_pos"][:2] - goal[:2]))
        had_contact = bool(np.any(records["grasp_contacts"]))
        if success:
            reason = "success"
        elif tracker.phase <= 1:
            reason = "unreachable_waypoint"
        elif max_lift < .06:
            reason = "grasp_failed" if not had_contact else "insufficient_lift"
        elif obs["object_pos"][2] < initial_object[2] + .03 and tracker.phase < 6:
            reason = "object_drop"
        else:
            reason = "horizon_timeout"
        metadata = {"schema_version": "1.0", "episode_id": f"{robot}_{teacher_origin}_{seed}",
                    "seed": int(seed), "mode": config.get("mode", "bootstrap"), "robot": robot,
                    "teacher_origin": teacher_origin, "source_policy_type": "scripted" if robot == "Panda" else None,
                    "source_demo_id": source_trace.get("metadata",{}).get("demo_id") if source_trace else None,
                    "success": success, "native_task_success": success, "reason_code": reason,
                    "placement_error_m": error, "max_lift_m": max_lift, "had_grasp_contact": had_contact,
                    "collision":bool(np.any(records["arm_collisions"])),"collision_definition":"robot arm external contact; expected gripper-object contacts excluded",
                    "collision_pairs":sorted(collision_pairs),"min_joint_limit_margin_rad":float(np.min(records["joint_limit_margins_rad"])),
                    "gripper_action_layout_verified":True,
                    "final_object_position": obs["object_pos"].tolist(), "goal_position": goal.tolist(),
                    "final_phase": PHASES[tracker.phase], "steps": len(records["actions"]),
                    "duration_seconds": time.monotonic()-started,
                    "privileged_state_used": True, "observation_mode": "oracle_sim_state",
                    "diagnostic_oracle_used": True, "frame": "target_world" if robot=="UR5e" else "source_world",
                    "units": {"position": "m", "timestamps": "s", "angles": "rad"},
                    "controller": adapter.description(), "feature_names": FEATURE_NAMES,
                    "feature_dim": len(FEATURE_NAMES), "transfer_parameters": {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in params.items()},
                    "render_error": render_error}
        arrays = {k: np.asarray(v) for k,v in records.items()}
        arrays["goal_position"] = goal
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        if video_path and frames:
            from ..viz.render_rollouts import write_video
            write_video(frames, video_path, fps=float(config.get("control_hz",20))/int(config.get("render_stride",2)),
                        label=f"{robot} | {teacher_origin} | success={success} | oracle sim state")
            metadata["video_path"] = str(video_path)
            arrays["metadata_json"] = np.array(json.dumps(metadata))
        return {**arrays, "metadata": metadata, "metrics": metadata}
    finally:
        env.close()


def collect_episode(config, seed, teacher_origin="target_scripted_bootstrap", source_trace=None, video_path=None, robot="UR5e"):
    return _episode(config, seed, robot, teacher_origin, source_trace=source_trace, video_path=video_path)


def run_policy_episode(config, policy_callable, seed, video_path=None):
    return _episode(config, seed, "UR5e", "learned_policy", policy_callable=policy_callable, video_path=video_path)


def save_episode(result, path):
    from ..schemas import validate_trajectory_arrays
    validate_trajectory_arrays(result["features"],result["actions"],result["timestamps"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k:v for k,v in result.items() if k not in ("metadata", "metrics")})
    return {**result["metadata"], "path": str(path)}
