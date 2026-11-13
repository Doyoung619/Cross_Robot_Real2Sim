"""Offline source-to-target kinematics. Never calls env.step or mj_step."""
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from vla_pilot_common import MATERIALS, DATASET, make_env, write_json


def resolve_xml(xml):
    root = ET.fromstring(xml)
    for node in root.iter():
        old = node.get("file")
        if old is None:
            continue
        marker = "/robosuite/models/assets/"
        if marker in old:
            target = MATERIALS / "sim_deps/robosuite/models/assets" / old.split(marker, 1)[1]
        elif "/assets/" in old:
            target = MATERIALS / "assets" / old.split("/assets/", 1)[1]
        else:
            raise ValueError(f"Unknown public source asset path: {old}")
        target = target.resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        node.set("file", str(target))
    return ET.tostring(root, encoding="unicode")


def source_model(group):
    model = mujoco.MjModel.from_xml_string(resolve_xml(group.attrs["model_file"]))
    data = mujoco.MjData(model)
    return model, data


def restore(model, data, state):
    if len(state) != 1 + model.nq + model.nv:
        raise ValueError("Source flattened-state contract mismatch")
    data.time = state[0]
    data.qpos[:] = state[1:1+model.nq]
    data.qvel[:] = state[1+model.nq:]
    mujoco.mj_forward(model, data)


def raw_target(env):
    return env.sim.model._model, env.sim.data._data


def copy_scene(source_m, source_d, target_m, target_d):
    """Copy recorded non-robot scene geometry/state, never target dynamics."""
    for sid in range(1, source_m.nbody):
        name = mujoco.mj_id2name(source_m, mujoco.mjtObj.mjOBJ_BODY, sid)
        if not name or name.startswith(("robot", "gripper", "mount")):
            continue
        tid = mujoco.mj_name2id(target_m, mujoco.mjtObj.mjOBJ_BODY, name)
        if tid >= 0:
            target_m.body_pos[tid] = source_m.body_pos[sid]
            target_m.body_quat[tid] = source_m.body_quat[sid]
    for sid in range(source_m.njnt):
        name = mujoco.mj_id2name(source_m, mujoco.mjtObj.mjOBJ_JOINT, sid)
        if not name or name.startswith(("robot", "gripper")):
            continue
        tid = mujoco.mj_name2id(target_m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if tid < 0 or source_m.jnt_type[sid] != target_m.jnt_type[tid]:
            continue
        n = {0: 7, 1: 4}.get(int(source_m.jnt_type[sid]), 1)
        sa, ta = source_m.jnt_qposadr[sid], target_m.jnt_qposadr[tid]
        target_d.qpos[ta:ta+n] = source_d.qpos[sa:sa+n]
    target_d.qvel[:] = 0
    target_d.qacc[:] = 0
    mujoco.mj_forward(target_m, target_d)


def ik(env, position, orientation, max_iterations=50):
    model, data = raw_target(env)
    robot = env.robots[0]
    qids = np.asarray(robot._ref_joint_pos_indexes)
    vids = np.asarray(robot._ref_joint_vel_indexes)
    jids = np.asarray(robot._ref_joint_indexes)
    lower, upper = model.jnt_range[jids].T
    site = robot.eef_site_id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for iteration in range(max_iterations):
        mujoco.mj_forward(model, data)
        ep = position-data.site_xpos[site]
        er = Rotation.from_matrix(orientation @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()
        if np.linalg.norm(ep) < .002 and np.linalg.norm(er) < .035:
            break
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jac = np.vstack([jp[:, vids], .2*jr[:, vids]])
        error = np.r_[ep, .2*er]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6)*1e-4, error)
        data.qpos[qids] = np.clip(data.qpos[qids]+np.clip(dq, -.18, .18), lower+.002, upper-.002)
    mujoco.mj_forward(model, data)
    ep = float(np.linalg.norm(position-data.site_xpos[site]))
    er = float(np.linalg.norm(Rotation.from_matrix(orientation @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()))
    return dict(position_error_m=ep, orientation_error_rad=er,
                feasible=ep < .015 and er < .15,
                joint_margin_rad=float(np.min(np.minimum(data.qpos[qids]-lower, upper-data.qpos[qids]))))


def unintended_contacts(env):
    model, data = raw_target(env)
    count = 0
    for contact in data.contact[:data.ncon]:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in (contact.geom1, contact.geom2)]
        if any(x.startswith("robot0_") for x in names) and any(not x.startswith(("robot", "gripper", "mount")) for x in names):
            if contact.dist < -.001:
                count += 1
    return count


def camera_calibration(env, source_m, source_d):
    """Fixed B wrist mount calibrated from one offline matched TCP pose."""
    model, data = raw_target(env)
    source_site = mujoco.mj_name2id(source_m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    source_hand = mujoco.mj_name2id(source_m, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
    env._pilot_state_rotation_offset = source_d.site_xmat[source_site].reshape(3,3).T @ source_d.xmat[source_hand].reshape(3,3)
    for name in ("agentview", "robot0_eye_in_hand"):
        sc = mujoco.mj_name2id(source_m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        tc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        parent = model.cam_bodyid[tc]
        parent_r = data.xmat[parent].reshape(3, 3)
        model.cam_pos[tc] = parent_r.T @ (source_d.cam_xpos[sc]-data.xpos[parent])
        local_r = parent_r.T @ source_d.cam_xmat[sc].reshape(3, 3)
        quat = Rotation.from_matrix(local_r).as_quat()
        model.cam_quat[tc] = quat[[3, 0, 1, 2]]
        model.cam_fovy[tc] = source_m.cam_fovy[sc]
    mujoco.mj_forward(model, data)


def canonical_state(env):
    model, data = raw_target(env)
    robot = env.robots[0]
    site = robot.eef_site_id
    p = data.site_xpos[site].copy()
    if hasattr(env, "_pilot_state_rotation_offset"):
        orientation = data.site_xmat[site].reshape(3,3) @ env._pilot_state_rotation_offset
    else:
        body = model.body(robot.robot_model.eef_name).id
        orientation = data.xmat[body].reshape(3,3)
    quat = Rotation.from_matrix(orientation).as_quat()
    # The downward-facing source poses use the positive-x quaternion branch,
    # including rotations slightly above pi. Do not introduce a sign jump at pi.
    if quat[0] < 0:
        quat *= -1
    denominator = np.sqrt(max(1-quat[3]**2, 0))
    angle = quat[:3]*(2*np.arccos(np.clip(quat[3],-1,1))/denominator) if denominator > 1e-8 else np.zeros(3)
    if len(robot._ref_gripper_joint_pos_indexes) == 2:
        grip = data.qpos[robot._ref_gripper_joint_pos_indexes].copy()
    else:
        geoms = robot.gripper.important_geoms
        left = model.geom(geoms["left_fingerpad"][0]).id
        right = model.geom(geoms["right_fingerpad"][0]).id
        half = np.clip(np.linalg.norm(data.geom_xpos[left]-data.geom_xpos[right])/2, 0, .04)
        grip = np.array([half, -half])
    return np.r_[p, angle, grip].astype(np.float32)


def dataset_pose(group, model, data, index):
    restore(model, data, group["states"][index])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()


def process(out, progress):
    started = time.monotonic()
    env = make_env("UR5e", 256)
    model, data = raw_target(env)
    default_qpos = data.qpos.copy()
    candidates = []
    try:
        with h5py.File(DATASET) as handle:
            names = sorted(handle["data"], key=lambda x: int(x.split("_")[-1]))
            train_names, eval_names = names[:-5], names[-5:-3]
            for number, name in enumerate(train_names):
                group = handle["data"][name]
                if float(group["rewards"][-1]) <= 0:
                    continue
                sm, sd = source_model(group)
                n = len(group["actions"])
                errors = []
                data.qpos[:] = default_qpos
                for index in np.linspace(0, n-1, 4, dtype=int):
                    p, rotation = dataset_pose(group, sm, sd, index)
                    copy_scene(sm, sd, model, data)
                    result = ik(env, p, rotation)
                    result["arm_collision_count"] = unintended_contacts(env)
                    errors.append(result)
                plate = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, "plate_1_main")
                restore(sm, sd, group["states"][0])
                candidates.append(dict(demo_id=name, steps=n,
                    initial_plate_xyz=sd.xpos[plate].tolist(), probes=errors,
                    max_position_error_m=max(x["position_error_m"] for x in errors),
                    feasible=all(x["feasible"] and x["arm_collision_count"] == 0 for x in errors)))
                if number % 5 == 0:
                    progress("selection", dict(inspected=number+1, total=len(train_names)))
                    write_json(out/"selection_candidates.json", candidates)
            eligible = sorted([x for x in candidates if x["feasible"]], key=lambda x: (x["steps"], x["demo_id"]))
            if not eligible:
                raise RuntimeError("No source episode passed B offline reachability/collision probes")
            chosen = [eligible.pop(0)]
            while eligible and len(chosen) < 3:
                best = max(eligible, key=lambda x: min(np.linalg.norm(np.array(x["initial_plate_xyz"])-y["initial_plate_xyz"]) for y in chosen))
                chosen.append(best)
                eligible.remove(best)
            selection = dict(candidate_count=len(candidates), eligible_count=sum(x["feasible"] for x in candidates),
                             selected=[x["demo_id"] for x in chosen], evaluation_source_ids=eval_names,
                             excluded_validation_ids=names[-5:], target_rollout_training_calls=0,
                             method="four_pose_B_IK_and_static_arm_collision_filter_then_shortest_and_farthest_plate_starts")
            write_json(out/"selection.json", selection)
            images, states, chunks, lineage, audits = [], [], [], [], []
            calibration = None
            for name in selection["selected"]:
                group = handle["data"][name]
                sm, sd = source_model(group)
                data.qpos[:] = default_qpos
                data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes] = 0  # static open Robotiq mechanism
                n = len(group["actions"])
                per_episode = []
                for index in range(0, n, 2):
                    p, rotation = dataset_pose(group, sm, sd, index)
                    copy_scene(sm, sd, model, data)
                    result = ik(env, p, rotation, max_iterations=35)
                    collisions = unintended_contacts(env)
                    audit = dict(demo_id=name, frame=index, **result, arm_collision_count=collisions)
                    audits.append(audit)
                    if not result["feasible"] or collisions:
                        continue
                    if calibration is None:
                        camera_calibration(env, sm, sd)
                        calibration = {"positions":model.cam_pos.copy(), "quaternions":model.cam_quat.copy(), "fovy":model.cam_fovy.copy(), "state_rotation_offset":env._pilot_state_rotation_offset}
                        np.savez(out/"camera_calibration.npz", **calibration)
                    pair = np.stack([env.sim.render(width=256,height=256,camera_name=camera) for camera in ("agentview","robot0_eye_in_hand")])
                    state = canonical_state(env)
                    action = np.asarray(group["actions"][index:min(index+50,n)], np.float32)
                    action = np.pad(action, ((0,50-len(action)),(0,0)), mode="edge")
                    # Re-express the original source OSC goal from matched B TCP.
                    action = action.copy()
                    action[0,:3] = np.clip(action[0,:3] + (p-state[:3])/.05, -1, 1)
                    images.append(pair)
                    states.append(state)
                    chunks.append(action)
                    lineage.append(dict(source_demo_id=name, source_frame=index, origin="source_derived_static_kinematic_render", executed_target_demo=False))
                    per_episode.append(index)
                progress("processing", dict(demo_id=name, accepted_frames=len(per_episode), total_frames=n))
            if len(states) < 20:
                raise RuntimeError(f"Only {len(states)} offline frames survived; insufficient processed source data")
            np.savez_compressed(out/"processed_source_data.npz", images=np.stack(images), states=np.stack(states), actions=np.stack(chunks))
            write_json(out/"processing_audit.json", dict(frames=audits, lineage=lineage,
                accepted_frames=len(states), image_shape=list(np.shape(images)), physics_steps_during_processing=0,
                target_executed_training_episodes=0, native_source_hz=20, anchor_stride=2, action_hz=20,
                pseudo_label_limit="kinematic conversion does not validate B contact dynamics", duration_seconds=time.monotonic()-started))
            return selection
    finally:
        env.close()
