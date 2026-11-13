#!/usr/bin/env python3
"""Inspect downloaded source data and choose ten task-matched source episodes."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/vla_readiness_20261113"


def qpos_address(xml, joint_name):
    offset = 0
    for element in ET.fromstring(xml).find("worldbody").iter():
        if element.tag not in ("joint", "freejoint"):
            continue
        if element.get("name") == joint_name:
            return offset
        typ = "free" if element.tag == "freejoint" else element.get("type", "hinge")
        offset += {"free": 7, "ball": 4}.get(typ, 1)
    raise ValueError(joint_name)


def main():
    path = next((OUT / "dataset/libero_spatial").glob("*.hdf5"))
    candidates = []
    with h5py.File(path) as f:
        for name, group in f["data"].items():
            actions = np.asarray(group["actions"])
            observations = group["obs"]
            success = float(group["rewards"][-1]) > 0
            valid = (len(actions) > 0 and actions.shape[1:] == (7,)
                     and np.isfinite(actions).all()
                     and all(len(value) == len(actions) for value in observations.values()))
            address = qpos_address(group.attrs["model_file"], "akita_black_bowl_1_joint0")
            xyz = np.asarray(group["states"][0, 1+address:1+address+3])
            candidates.append(dict(demo_id=name, steps=len(actions), success=success,
                                   consistent=bool(valid), initial_bowl_xyz=xyz.tolist()))
        task = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
    valid = sorted([x for x in candidates if x["success"] and x["consistent"]],
                   key=lambda x: (x["steps"], x["demo_id"]))
    selected, remaining = [valid[0]], valid[1:]
    while len(selected) < 10 and remaining:
        chosen = max(remaining, key=lambda x: min(
            np.linalg.norm(np.array(x["initial_bowl_xyz"])-y["initial_bowl_xyz"])
            for y in selected))
        selected.append(chosen)
        remaining.remove(chosen)
    manifest = json.loads((OUT / "download_manifest.json").read_text())
    result = dict(status="source_materials_ready_offline_B_conditioned_processing_not_implemented",
                  task=task, source_robot="Panda", target_robot="UR5e",
                  source_checkpoint="lerobot/smolvla_libero",
                  checkpoint_revision="a24f1f380461b665158d52acd140152c79e8b36d",
                  checkpoint_training_dataset="lerobot/libero",
                  source_dataset_path=str(path.relative_to(ROOT)),
                  source_demos=len(candidates), source_steps=sum(x["steps"] for x in candidates),
                  terminal_reward_successes=sum(x["success"] for x in candidates),
                  consistent_demos=sum(x["consistent"] for x in candidates),
                  selection="task-specific successful demos; shortest then max-min initial bowl xyz diversity",
                  selection_status="provisional_geometry_baseline_not_B_aware",
                  selected_source_demos=selected,
                  selected_source_steps=sum(x["steps"] for x in selected),
                  download_files=len(manifest["files"]), download_errors=manifest["errors"],
                  download_bytes=sum(x["bytes"] for x in manifest["files"]),
                  target_training_episodes=0,
                  target_executed_training_demonstrations_allowed=False,
                  target_simulator_rollout_training_allowed=False,
                  target_evaluation_feedback_for_selection_or_tuning_allowed=False,
                  allowed_target_knowledge=["robot_geometry", "kinematics", "joint_limits", "collision_geometry", "gripper_geometry", "control_contract"],
                  offline_target_pseudo_labels_generated=False,
                  source_policy_rollout_success="not_measured",
                  target_policy_success="not_measured")
    (OUT / "readiness.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
