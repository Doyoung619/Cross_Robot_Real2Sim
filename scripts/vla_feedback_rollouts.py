"""Real B policy rollouts and measured feedback diagnostics.

Target actions are retained for diagnosis only. This module does not create
teacher demonstrations or invoke a training update.
"""
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np
import torch

from vla_pilot_common import DATASET, make_env, write_json
from vla_pilot_processing import raw_target, canonical_state, unintended_contacts
from vla_pilot_evaluate import initial_scene, frame
from vla_pilot_learning import load_model, InputContract


def _contact_ids(env):
    model, _ = raw_target(env)
    plate, gripper = set(), set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith("plate_1") and (model.geom_contype[i] or model.geom_conaffinity[i]):
            plate.add(i)
        if name.startswith("gripper0_") and (model.geom_contype[i] or model.geom_conaffinity[i]):
            gripper.add(i)
    pad = env.robots[0].gripper.important_geoms
    pads = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
            for key in ("left_fingerpad", "right_fingerpad") for name in pad[key]]
    return plate, gripper, [i for i in pads if i >= 0]


def _contact_measurement(env, ids):
    model,data=raw_target(env)
    plate,gripper,pads=ids
    contacts=[];force=0.
    for index,contact in enumerate(data.contact[:data.ncon]):
        a,b=int(contact.geom1),int(contact.geom2)
        if (a in plate and b in gripper) or (b in plate and a in gripper):
            value=np.zeros(6)
            mujoco.mj_contactForce(model,data,index,value)
            force+=abs(float(value[0]))
            contacts.append(dict(geom1=model.geom(a).name,geom2=model.geom(b).name,
                                 distance_m=float(contact.dist),normal_force_N=abs(float(value[0]))))
    best=.3;vector=np.zeros(3);points=np.zeros(6)
    for a in pads:
        for b in plate:
            candidate=np.zeros(6)
            try:
                distance=float(mujoco.mj_geomDistance(model,data,a,b,.3,candidate))
            except (TypeError,ValueError):
                distance=float(np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b]))
                candidate[:3]=data.geom_xpos[a];candidate[3:]=data.geom_xpos[b]
            if distance<best:
                best=distance;points=candidate.copy();vector=candidate[3:]-candidate[:3]
    return dict(gripper_plate_contact=bool(contacts),plate_contact_count=len(contacts),
                contact_normal_force_N=force,min_gripper_plate_distance_m=best,
                closest_surface_vector_m=vector.tolist(),closest_surface_points=points.tolist(),
                contacts=contacts)


def _aggregate(trace, initial_plate):
    contacts=np.array([x["gripper_plate_contact"] for x in trace],bool)
    distances=np.array([x["min_gripper_plate_distance_m"] for x in trace])
    eef_gap=np.array([np.linalg.norm(np.array(x["eef_xyz"])-x["plate_xyz"]) for x in trace])
    near=np.argsort(distances)[:max(5,len(trace)//5)]
    heights=np.array([x["eef_xyz"][2]-x["plate_xyz"][2] for x in trace])
    directions=[x["push_direction_alignment"] for x in trace if x["push_direction_alignment"] is not None and x["min_gripper_plate_distance_m"]<.08]
    gains=[x["plate_response_gain"] for x in trace if x["gripper_plate_contact"] and x["plate_response_gain"] is not None]
    errors=np.array([x["goal_error_m"] for x in trace])
    stall_window=min(40,len(trace))
    return dict(contact_steps=int(contacts.sum()),contact_fraction=float(contacts.mean()),
        plate_contact_fraction=float(contacts.mean()),contact_rate=float(contacts.mean()),
        max_contact_normal_force_N=max(x["contact_normal_force_N"] for x in trace),
        min_gripper_plate_distance_m=float(distances.min()),
        mean_near_surface_gap_m=float(distances[near].mean()),
        mean_eef_plate_gap_m=float(eef_gap[near].mean()),
        mean_near_plate_height_m=float(heights[near].mean()),
        mean_closest_surface_vector_m=np.mean([trace[i]["closest_surface_vector_m"] for i in near],axis=0).tolist(),
        mean_push_direction_alignment=float(np.mean(directions)) if directions else None,
        direction_alignment=float(np.mean(directions)) if directions else None,
        response_gain=float(np.mean(gains)) if gains else None,
        best_goal_error_m=float(errors.min()),
        final_window_progress_m=float(errors[-stall_window]-errors[-1]),
        max_plate_displacement_m=max(float(np.linalg.norm(np.array(x["plate_xyz"])-initial_plate)) for x in trace))


def evaluate_feedback(out, checkpoint, stage_label, source_ids, progress, deadline, horizon=220):
    """Evaluate one checkpoint with rich development/final traces and video."""
    import imageio.v2 as imageio
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/"videos").mkdir(exist_ok=True)
    policy=load_model(adapter=True,adapter_path=checkpoint)
    contract=InputContract(policy)
    env=make_env("UR5e",256)
    rows=[]
    try:
        with h5py.File(DATASET) as handle:
            for case_index,demo in enumerate(source_ids):
                if time.time()>deadline-60:raise TimeoutError("Rollout deadline with reporting reserve")
                torch.manual_seed(1000+case_index)
                initial_scene(env,handle["data"][demo],"UR5e",out)
                policy.reset();model,data=raw_target(env)
                plate=model.body("plate_1_main").id
                goal=model.site("main_table_stove_front_region").id
                initial_plate=data.xpos[plate].copy();initial_error=float(np.linalg.norm(initial_plate[:2]-data.site_xpos[goal,:2]))
                initial_state=canonical_state(env).copy();ids=_contact_ids(env)
                video=out/"videos"/f"{stage_label}_{demo}.mp4"
                trace=[];hold=0;ever=False;collisions=0;started=time.monotonic()
                trace_path=out/f"{stage_label}_{demo}_trace.json"
                with imageio.get_writer(str(video),fps=10,codec="libx264",quality=7,macro_block_size=1) as writer:
                    for step in range(horizon):
                        previous_plate=data.xpos[plate].copy()
                        pair=np.stack([env.sim.render(width=256,height=256,camera_name=camera) for camera in ("agentview","robot0_eye_in_hand")])
                        batch=contract.batch(pair[None],canonical_state(env)[None])
                        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                            prediction=policy.select_action(batch)
                        raw_action=contract.decode(prediction)[0]
                        if not np.isfinite(raw_action).all():raise RuntimeError("Non-finite VLA action")
                        action=np.clip(raw_action,-1,1)
                        env.step(action)
                        success=bool(env.check_success());ever|=success;hold=hold+1 if success else 0
                        collisions+=unintended_contacts(env)>0
                        p=data.xpos[plate].copy();g=data.site_xpos[goal].copy();eef=canonical_state(env)
                        error=float(np.linalg.norm(p[:2]-g[:2]));direction=g[:2]-p[:2]
                        norm=float(np.linalg.norm(action[:2])*np.linalg.norm(direction))
                        alignment=float(np.dot(action[:2],direction)/norm) if norm>1e-6 else None
                        command_length=float(np.linalg.norm(action[:2])*.05)
                        response=float(np.linalg.norm(p[:2]-previous_plate[:2])/command_length) if command_length>1e-4 else None
                        entry=dict(step=step,time_s=step/20,eef_xyz=eef[:3].tolist(),eef_axisangle=eef[3:6].tolist(),
                            eef_site_quaternion_wxyz=Rotation_from_site(model,data,env.robots[0].eef_site_id),
                            plate_xyz=p.tolist(),goal_xyz=g.tolist(),action=action.tolist(),raw_action=raw_action.tolist(),
                            goal_error_m=error,push_direction_alignment=alignment,plate_response_gain=response,
                            native_success=success,**_contact_measurement(env,ids))
                        trace.append(entry)
                        if step%2==0:writer.append_data(frame(env,stage_label,demo,step,success,error))
                        if step%50==0:
                            progress("feedback_rollout",dict(stage=stage_label,demo=demo,step=step,goal_error_m=error,contact=entry["gripper_plate_contact"]))
                            write_json(trace_path,dict(stage=stage_label,source_initial_scene_id=demo,training_usage="diagnostic_only_not_action_supervision",steps=trace))
                        if hold>=20:break
                        if time.time()>deadline-60:raise TimeoutError("Rollout deadline reached")
                aggregate=_aggregate(trace,initial_plate)
                remaining=data.site_xpos[goal,:2]-data.xpos[plate,:2]
                row=dict(condition=stage_label,robot="UR5e",source_initial_scene_id=demo,success=bool(env.check_success()),
                    ever_success=ever,stable_success_steps=hold,steps=len(trace),goal_error_m=trace[-1]["goal_error_m"],
                    initial_goal_error_m=initial_error,goal_progress_m=initial_error-trace[-1]["goal_error_m"],
                    plate_displacement_m=float(np.linalg.norm(data.xpos[plate]-initial_plate)),
                    arm_collision_steps=int(collisions),initial_canonical_state=initial_state.tolist(),
                    initial_plate_position=initial_plate.tolist(),final_plate_position=data.xpos[plate].tolist(),
                    goal_position=data.site_xpos[goal].tolist(),remaining_goal_vector_xy=remaining.tolist(),
                    video=str(video.relative_to(out)),trace=str(trace_path.relative_to(out)),
                    duration_seconds=time.monotonic()-started,training_usage="diagnostic_only_not_action_supervision",**aggregate)
                rows.append(row)
                write_json(trace_path,dict(stage=stage_label,source_initial_scene_id=demo,training_usage="diagnostic_only_not_action_supervision",steps=trace,summary=row))
                write_json(out/"evaluation.json",rows)
                progress("feedback_rollout_completed",row)
    finally:
        env.close();del policy,contract;torch.cuda.empty_cache()
    return rows


def Rotation_from_site(model,data,site):
    quaternion=np.empty(4)
    mujoco.mju_mat2Quat(quaternion,data.site_xmat[site])
    return quaternion.tolist()


def diagnose(rows):
    """Measured heuristic candidate diagnosis, not a causal root-cause proof."""
    failed=[r for r in rows if not r["success"]]
    selected=failed or rows
    if not selected:raise ValueError("No rollouts to diagnose")
    mean=lambda key:float(np.mean([r[key] for r in selected]))
    contacts=mean("contact_fraction")
    alignments=[r["direction_alignment"] for r in selected if r["direction_alignment"] is not None]
    alignment=float(np.mean(alignments)) if alignments else None
    gains=[r["response_gain"] for r in selected if r["response_gain"] is not None]
    gain=float(np.mean(gains)) if gains else None
    gap=mean("mean_near_surface_gap_m")
    if contacts<.03 and gap>.003:
        hypothesis="insufficient_plate_contact"
    elif alignment is not None and alignment<.45:
        hypothesis="push_direction_misaligned"
    elif contacts>.05 and (gain is not None and gain<.15):
        hypothesis="contact_without_sufficient_plate_response"
    elif mean("final_window_progress_m")<.005:
        hypothesis="early_progress_then_stall"
    else:
        hypothesis="incomplete_goal_progress"
    vector=np.mean([r["mean_closest_surface_vector_m"] for r in selected],axis=0)
    remaining=np.mean([r["remaining_goal_vector_xy"] for r in selected],axis=0)
    request=dict(source_query="successful contact-rich A segments aligned with remaining goal direction",
                 desired_push_direction_xy=(remaining/max(np.linalg.norm(remaining),1e-8)).tolist(),
                 measured_contact_gap_vector_m=vector.tolist(),
                 max_contact_position_correction_m=.02,max_height_correction_m=.01,
                 require_recomputed_full_action_chunks=True,
                 target_trace_action_supervision_allowed=False)
    return dict(status="diagnosed",classification="heuristic_candidates_not_causal_proof",
        evaluated_episodes=len(rows),failed_episodes=len(failed),dominant_hypothesis=hypothesis,
        contact_fraction=contacts,plate_contact_fraction=contacts,contact_rate=contacts,
        mean_eef_plate_gap_m=mean("mean_eef_plate_gap_m"),mean_near_surface_gap_m=gap,
        mean_near_plate_height_m=mean("mean_near_plate_height_m"),
        mean_closest_surface_vector_m=vector.tolist(),mean_remaining_goal_vector_xy=remaining.tolist(),
        direction_alignment=alignment,response_gain=gain,mean_goal_error_m=mean("goal_error_m"),
        mean_plate_displacement_m=mean("plate_displacement_m"),
        feedback_request=request,
        hypotheses=[dict(name="contact_geometry",evidence=dict(contact_fraction=contacts,near_surface_gap_m=gap)),
                    dict(name="push_direction",evidence=dict(alignment=alignment)),
                    dict(name="contact_dynamics_or_action_scale",evidence=dict(response_gain=gain),causally_verified=False)])
