"""Isolated local runtime for the one-hour source-only VLA pilot."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "runs/vla_readiness_20261113"
DATASET = MATERIALS / "dataset/libero_goal/push_the_plate_to_the_front_of_the_stove_demo.hdf5"
TASK = "push the plate to the front of the stove"


def bootstrap():
    for path in (MATERIALS / "deps", MATERIALS / "hf_libero-0.1.3", MATERIALS / "sim_deps"):
        sys.path.insert(0, str(path))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_HUB_OFFLINE"] = "1"
    cfg = MATERIALS / "pilot_libero_config"
    cfg.mkdir(exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    base = MATERIALS / "hf_libero-0.1.3/libero/libero"
    import yaml
    (cfg / "config.yaml").write_text(yaml.safe_dump(dict(
        benchmark_root=str(base), bddl_files=str(base / "bddl_files"),
        init_states=str(base / "init_files"), datasets=str(MATERIALS / "dataset"),
        assets=str(MATERIALS / "assets"))))


def make_env(robot="Panda", size=256):
    import libero.libero as settings
    settings._assets_path_cache = str(MATERIALS / "assets")
    from robosuite.models.robots.manipulators.ur5e_robot import UR5e
    if not getattr(UR5e, "_pilot_table_compatibility", False):
        original = UR5e.base_xpos_offset.fget
        def offsets(self):
            return {**original(self), "kitchen_table": lambda length: (-.16-length/2, 0, 0)}
        UR5e.base_xpos_offset = property(offsets)
        UR5e._pilot_table_compatibility = True
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm
    if "MountedUR5e" not in ROBOT_CLASS_MAPPING:
        # LIBERO's tabletop task prepends Mounted to every robot name.
        # Preserve the UR5e geometry and its default RethinkMount/gripper.
        class MountedUR5e(UR5e):
            pass
        ROBOT_CLASS_MAPPING["MountedUR5e"] = SingleArm
    bddl = MATERIALS / "hf_libero-0.1.3/libero/libero/bddl_files/libero_goal/push_the_plate_to_the_front_of_the_stove.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), robots=[robot],
                           camera_heights=size, camera_widths=size,
                           control_freq=20, horizon=300, ignore_done=True,
                           hard_reset=False, render_gpu_device_id=0)
    env.reset()
    return env


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    temp.replace(path)


if __name__ == "__main__":
    bootstrap()
    import numpy as np
    for robot in ("Panda", "UR5e"):
        env = make_env(robot, size=128)
        try:
            obs = env.env._get_observations(force_update=True)
            print(robot, "nq/nv", env.sim.model.nq, env.sim.model.nv,
                  "eef", env.robots[0].eef_site_id,
                  "qpos", env.robots[0]._ref_joint_pos_indexes,
                  "keys", list(obs), flush=True)
            print("controller", env.robots[0].controller.name,
                  "initial_eef", obs["robot0_eef_pos"], flush=True)
            result = env.step(np.r_[np.zeros(6), -1.])
            print(robot, "step/render OK", result[0]["agentview_image"].shape, flush=True)
        finally:
            env.close()
