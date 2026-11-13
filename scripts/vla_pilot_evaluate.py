"""Closed-loop final evaluation only. Its outputs are never training inputs."""
import json
import time

import h5py
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from vla_pilot_common import DATASET, TASK, make_env, write_json
from vla_pilot_processing import (source_model, restore, dataset_pose, resolve_xml,
    raw_target, copy_scene, ik, canonical_state, unintended_contacts)
from vla_pilot_learning import load_model, InputContract


def initial_scene(env, group, robot, out):
    sm, sd = source_model(group)
    p, rotation = dataset_pose(group, sm, sd, 0)
    if robot == "Panda":
        env.reset_from_xml_string(resolve_xml(group.attrs["model_file"]))
        env.set_init_state(np.asarray(group["states"][0]))
    else:
        env.reset()
        model, data = raw_target(env)
        copy_scene(sm, sd, model, data)
        data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes] = 0
        result = ik(env, p, rotation, max_iterations=100)
        if not result["feasible"]:
            raise RuntimeError(f"Evaluation initial B pose infeasible: {result}")
        with np.load(out/"camera_calibration.npz") as camera:
            model.cam_pos[:] = camera["positions"]
            model.cam_quat[:] = camera["quaternions"]
            model.cam_fovy[:] = camera["fovy"]
            env._pilot_state_rotation_offset = camera["state_rotation_offset"].copy()
        data.qvel[:] = 0
        mujoco.mj_forward(model,data)
    controller = env.robots[0].controller
    controller.update(force=True)
    controller.update_initial_joints(env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy())
    controller.reset_goal()
    env.robots[0].gripper.current_action = np.array([1.,-1.]) if robot == "Panda" else np.array([-1.])
    return p


def frame(env, condition, demo, step, success, distance):
    image = Image.new("RGB",(640,360),(12,20,32))
    draw = ImageDraw.Draw(image)
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font=ImageFont.truetype(font_path,17)
    small=ImageFont.truetype(font_path,13)
    for i,camera in enumerate(("agentview","robot0_eye_in_hand")):
        raw = env.sim.render(width=256,height=256,camera_name=camera)
        image.paste(Image.fromarray(raw[::-1].copy()),(20+i*310,64))
    draw.text((15,8),condition,font=font,fill="white")
    draw.text((15,33),"Task: push the plate to the front of the stove",font=small,fill=(180,210,240))
    draw.text((20,325),f"Held-out {demo} | t={step/20:.2f}s | success={success} | goal error={distance:.3f}m",font=small,fill="white")
    return np.asarray(image)


def evaluate(out, progress, deadline, horizon=200, conditions=None):
    import imageio.v2 as imageio
    selection=json.loads((out/"selection.json").read_text())
    cases=selection["evaluation_source_ids"]
    conditions=conditions or [("A_pretrained_Panda","Panda",False),
                ("B_pretrained_UR5e","UR5e",False),
                ("B_adapted_UR5e","UR5e",True)]
    rows=[]
    (out/"videos").mkdir(exist_ok=True)
    for entry in conditions:
        condition,robot,adapted=entry[:3]
        checkpoint=entry[3] if len(entry)>3 else (out/"adapter.safetensors" if adapted else None)
        policy=load_model(adapter=adapted,adapter_path=checkpoint)
        contract=InputContract(policy)
        env=make_env(robot,256)
        try:
            with h5py.File(DATASET) as handle:
                for case_index,demo in enumerate(cases):
                    if time.time()>deadline-90:
                        raise TimeoutError("Stopping evaluation with time reserved for final report")
                    torch.manual_seed(1000+case_index)
                    group=handle["data"][demo]
                    initial_scene(env,group,robot,out)
                    policy.reset()
                    model,data=raw_target(env)
                    plate=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"plate_1_main")
                    goal=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"main_table_stove_front_region")
                    initial_plate=data.xpos[plate].copy()
                    initial_state=canonical_state(env)
                    actions=[];collisions=0;ever=False;hold=0;last_error=None
                    video=out/"videos"/f"{condition}_{demo}.mp4"
                    started=time.monotonic()
                    with imageio.get_writer(str(video),fps=10,codec="libx264",quality=7,macro_block_size=1) as writer:
                        for step in range(horizon):
                            pair=np.stack([env.sim.render(width=256,height=256,camera_name=camera) for camera in ("agentview","robot0_eye_in_hand")])
                            state=canonical_state(env)
                            batch=contract.batch(pair[None],state[None])
                            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                                value=policy.select_action(batch)
                            raw_action=contract.decode(value)[0]
                            if not np.isfinite(raw_action).all():
                                raise RuntimeError("Nonfinite VLA action")
                            action=np.clip(raw_action,-1,1)
                            env.step(action)
                            actions.append(action.tolist())
                            success=bool(env.check_success())
                            ever |= success
                            hold=hold+1 if success else 0
                            collisions += unintended_contacts(env)>0
                            last_error=float(np.linalg.norm(data.xpos[plate,:2]-data.site_xpos[goal,:2]))
                            if step%2==0:
                                writer.append_data(frame(env,condition,demo,step,success,last_error))
                            if step%50==0:
                                progress("evaluation",dict(condition=condition,demo=demo,step=step,success=success,goal_error_m=last_error))
                            if hold>=20:
                                break
                            if time.time()>deadline-90:
                                raise TimeoutError("Evaluation deadline reached")
                    row=dict(condition=condition,robot=robot,source_initial_scene_id=demo,success=bool(env.check_success()),
                             ever_success=ever,stable_success_steps=hold,steps=len(actions),goal_error_m=last_error,
                             plate_displacement_m=float(np.linalg.norm(data.xpos[plate]-initial_plate)),
                             arm_collision_steps=int(collisions),initial_canonical_state=initial_state.tolist(),
                             initial_plate_position=initial_plate.tolist(),final_plate_position=data.xpos[plate].tolist(),
                             goal_position=data.site_xpos[goal].tolist(),video=str(video.relative_to(out)),
                             duration_seconds=time.monotonic()-started,training_usage="none_final_evaluation_only",
                             native_success_predicate="LIBERO On plate_1 main_table_stove_front_region")
                    rows.append(row)
                    write_json(out/"evaluation.json",rows)
                    write_json(out/f"{condition}_{demo}_actions.json",actions)
                    progress("evaluation_episode_completed",row)
        finally:
            env.close()
            del policy
            torch.cuda.empty_cache()
    return rows
