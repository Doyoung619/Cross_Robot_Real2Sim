"""Installed-version-tested headless PickPlaceCan adapter (SI units)."""
from __future__ import annotations

import os
import numpy as np


def make_env(config, robot="UR5e", seed=42):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    np.random.seed(int(seed))
    controller = load_composite_controller_config(controller="BASIC")
    env = suite.make("PickPlaceCan", robots=robot, controller_configs=controller,
                     has_renderer=False, has_offscreen_renderer=True,
                     use_camera_obs=False, use_object_obs=True,
                     initialization_noise=None, control_freq=int(config.get("control_hz", 20)),
                     horizon=int(config.get("horizon", 300)), ignore_done=True,
                     hard_reset=False, reward_shaping=True, seed=int(seed))
    env.reset()
    env._crossrobot_initial_height = float(observation(env)["object_pos"][2])
    return env


def object_index(env):
    return next(i for i, obj in enumerate(env.objects) if "can" in obj.name.lower())


def observation(env):
    obs = env._get_observations(force_update=True)
    idx = object_index(env)
    obj = env.objects[idx]
    obj_pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id[obj.name]], dtype=float).copy()
    robot = env.robots[0]
    gripper = robot.gripper[robot.arms[0]]
    contact = bool(env._check_grasp(gripper=gripper, object_geoms=obj.contact_geoms))
    qpos = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=float)
    return {"eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=float),
            "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=float),
            "object_pos": obj_pos, "goal_pos": np.asarray(env.target_bin_placements[idx], dtype=float).copy(),
            "gripper_opening": float(np.mean(np.abs(qpos))) if len(qpos) else 0.0,
            "grasp_contact": contact, "raw": obs}


def goal_success(env, obs=None):
    """Use the actual task's position-and-release predicate, not video geometry."""
    return bool(env._check_success())


def safety_diagnostics(env):
    """Ground-truth arm contacts and joint-limit margins, used only in reports."""
    robot=env.robots[0]
    arm_geoms=set(robot.robot_model.contact_geoms)
    pairs=[]
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        name1=env.sim.model.geom_id2name(contact.geom1) or "unnamed"
        name2=env.sim.model.geom_id2name(contact.geom2) or "unnamed"
        if (name1 in arm_geoms and name2 not in arm_geoms and not name2.startswith("gripper")) or (name2 in arm_geoms and name1 not in arm_geoms and not name1.startswith("gripper")):
            pairs.append(f"{name1}|{name2}")
    q=np.asarray(env.sim.data.qpos[robot._ref_joint_pos_indexes])
    limits=np.asarray(env.sim.model.jnt_range[robot._ref_joint_indexes])
    valid=limits[:,0]!=limits[:,1]
    margin=float(np.min(np.minimum(q[valid]-limits[valid,0],limits[valid,1]-q[valid]))) if np.any(valid) else float("inf")
    return {"arm_collision":bool(pairs),"arm_collision_pairs":pairs,"joint_limit_margin_rad":margin}


def render_frame(env, config):
    return np.flipud(env.sim.render(width=int(config.get("render_width",224)),
                                  height=int(config.get("render_height",224)),
                                  camera_name="agentview")).copy()


def smoke_test(config, output_dir=None):
    from .robot_action_adapter import RobotActionAdapter
    import importlib.metadata
    results = {}
    for robot in ("Panda", "UR5e"):
        env = make_env(config, robot, 42)
        try:
            adapter = RobotActionAdapter(env)
            env.robots[0].print_action_info()
            frames = []
            for _ in range(3):
                env.step(adapter.to_env([0, 0, 0, -1]))
                frames.append(render_frame(env, config))
            results[robot] = {"steps": 3, "rendered_frames": len(frames),
                              "frame_shape": list(frames[0].shape), **adapter.description(),
                              "print_action_info_checked":True,"create_action_vector_checked":True,
                              "observation_keys": list(observation(env)["raw"])}
            if output_dir:
                from pathlib import Path
                import imageio.v2 as imageio
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(str(Path(output_dir)/f"smoke_{robot}.mp4"), frames, fps=20)
        finally:
            env.close()
    return {"robots": results, "robosuite": importlib.metadata.version("robosuite"),
            "mujoco": importlib.metadata.version("mujoco"), "render_backend": os.getenv("MUJOCO_GL")}
