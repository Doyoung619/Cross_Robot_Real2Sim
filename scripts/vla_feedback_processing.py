"""Failure-conditioned source selection and offline contact-geometry retargeting.

Target rollout summaries may choose source examples and a bounded geometric
correction. No target trajectory becomes an action demonstration, and no target
physics step is executed in this module. Static distances do not prove dynamics.
"""
from pathlib import Path
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from vla_pilot_common import DATASET, make_env, write_json
from vla_pilot_processing import (source_model, restore, dataset_pose, raw_target,
    copy_scene, ik, canonical_state, unintended_contacts)


def _joint_address(xml, name):
    address=0
    for node in ET.fromstring(xml).find('worldbody').iter():
        if node.tag not in ('joint','freejoint'):continue
        if node.get('name')==name:return address
        typ='free' if node.tag=='freejoint' else node.get('type','hinge')
        address+={'free':7,'ball':4}.get(typ,1)
    raise ValueError(name)


def _summary(diagnostic):
    """Accept root diagnostics as either a flat summary or a nested summary."""
    d=dict(diagnostic.get('summary',{}));d.update(diagnostic)
    vec=d.get('mean_remaining_goal_vector_xy',d.get('remaining_goal_vector_xy',d.get('goal_direction_xy',[-.1,.2])))
    vec=np.asarray(vec,dtype=float).reshape(-1)[:2]
    if len(vec)!=2 or not np.isfinite(vec).all() or np.linalg.norm(vec)<1e-8:vec=np.array([-.1,.2])
    direction=vec/np.linalg.norm(vec)
    fraction=float(d.get('contact_fraction',d.get('plate_contact_fraction',0)) or 0.)
    gain_value=d.get('response_gain',.25)
    gain=float(gain_value) if gain_value is not None else .25
    if not np.isfinite(gain) or gain<=0:gain=.25
    return d,direction,fraction,int(np.clip(round(1/gain),2,5))


def _features(group):
    n=len(group['actions']);a=_joint_address(group.attrs['model_file'],'plate_1_joint0')
    plate=np.asarray(group['states'][:,1+a:1+a+3]);eef=np.asarray(group['obs/ee_pos'])
    speed=np.linalg.norm(np.diff(plate[:,:2],axis=0,prepend=plate[:1,:2]),axis=1)
    moving=np.flatnonzero(speed>3e-4)
    lo=max(0,int(moving[0])-8) if len(moving) else n//3
    hi=min(n-1,int(moving[-1])+5) if len(moving) else 2*n//3
    delta=plate[-1,:2]-plate[0,:2];direction=delta/max(np.linalg.norm(delta),1e-8)
    approach=eef[lo,:2]-plate[lo,:2]
    return dict(steps=n,plate=plate,eef=eef,lo=lo,hi=hi,direction=direction,
                displacement=float(np.linalg.norm(delta)),approach=approach,
                contact_window_steps=hi-lo+1,initial_plate_xyz=plate[0].tolist())


def _plate_geoms(model):
    ids=[]
    for i in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        if name.startswith('plate_1_') and (model.geom_contype[i] or model.geom_conaffinity[i]):ids.append(i)
    if not ids:raise ValueError('No plate collision geometry')
    return ids


def _finger_distance(env):
    model,data=raw_target(env);plate=_plate_geoms(model)
    fingers=[model.geom(name).id for key in ('left_fingerpad','right_fingerpad') for name in env.robots[0].gripper.important_geoms[key]]
    distances=[];points=np.zeros(6)
    for finger in fingers:
        for geom in plate:
            distances.append(float(mujoco.mj_geomDistance(model,data,finger,geom,1.,points)))
    return min(distances)


def _blend(index,lo,hi):
    # Smooth bounded correction; no instantaneous joint teleport between stages.
    return float(min(np.clip((index-lo+6)/6,0,1),np.clip((hi+6-index)/6,0,1)))


def _optimize_offset(env,group,features,round_index,direction,contact_fraction,diagnostic=None):
    sm,sd=source_model(group);model,data=raw_target(env);initial=data.qpos.copy()
    radius=min(.008+.004*round_index,.02)
    along=np.r_[direction,0.];across=np.array([-direction[1],direction[0],0.])
    offsets=[np.zeros(3),np.array([0,0,-min(radius/2,.01)]),np.array([0,0,-min(radius,.01)]),
             along*radius/2,along*radius,across*radius/2,-across*radius/2,
             along*radius/2+np.array([0,0,-radius/2])]
    gap=np.asarray((diagnostic or {}).get('mean_closest_surface_vector_m',[0,0,0]),dtype=float)
    if gap.shape==(3,) and np.isfinite(gap).all() and np.linalg.norm(gap)>1e-5:
        gap=gap*min(1.,.02/np.linalg.norm(gap));gap[2]=np.clip(gap[2],-.01,.01)
        offsets.append(gap)
    indices=np.unique(np.linspace(features['lo'],features['hi'],3,dtype=int))
    trials=[]
    for offset in offsets:
        data.qpos[:]=initial;errors=[];angular_errors=[];distances=[];collisions=[]
        for index in indices:
            p,r=dataset_pose(group,sm,sd,int(index));copy_scene(sm,sd,model,data)
            result=ik(env,p+offset,r,max_iterations=45)
            distance=_finger_distance(env);collision=unintended_contacts(env)
            distances.append(distance);errors.append(result['position_error_m']);angular_errors.append(result['orientation_error_rad']);collisions.append(collision)
        # Near contact is rewarded; large penetration, infeasible IK and arm
        # collisions are penalized. This is a static geometry objective only.
        d=np.asarray(distances)
        cost=float(np.mean(np.abs(d-.001)+4*np.maximum(-d-.003,0))+
                   2*np.mean(errors)+.002*np.linalg.norm(offset)/.02+.03*sum(collisions))
        trials.append(dict(offset_m=offset.tolist(),static_objective=cost,
                           finger_plate_signed_distances_m=distances,
                           max_ik_error_m=max(errors),max_orientation_error_rad=max(angular_errors),arm_collisions=sum(collisions)))
    eligible=[x for x in trials if x['max_ik_error_m']<.015 and x['max_orientation_error_rad']<.15 and not x['arm_collisions']]
    best=min(eligible or trials,key=lambda x:x['static_objective'])
    return best,trials


def process_feedback(out,parent,diagnostic,round_index,progress):
    """Append new A-derived corrective windows to the unchanged parent data.

    Returns new_rows_start/new_rows_count and writes an optional actions_id_pad
    mask. Caller should pass that mask to SmolVLA's training forward method.
    """
    out=Path(out);parent=Path(parent);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();summary,direction,contact_fraction,lookahead=_summary(diagnostic)
    old_selection=json.loads((parent/'selection.json').read_text())
    used=set(old_selection.get('selected',[]))|set(old_selection.get('feedback_selected',[]))
    with np.load(parent/'processed_source_data.npz') as old:
        old_images=old['images'].copy();old_states=old['states'].copy();old_actions=old['actions'].copy()
        old_mask=old['actions_id_pad'].copy() if 'actions_id_pad' in old else np.zeros(old_actions.shape[:2],bool)
    shutil.copyfile(parent/'camera_calibration.npz',out/'camera_calibration.npz')
    env=make_env('UR5e',256);model,data=raw_target(env);default=data.qpos.copy()
    with np.load(out/'camera_calibration.npz') as camera:
        model.cam_pos[:]=camera['positions'];model.cam_quat[:]=camera['quaternions'];model.cam_fovy[:]=camera['fovy']
        env._pilot_state_rotation_offset=camera['state_rotation_offset'].copy()
    candidates=[];images=[];states=[];chunks=[];masks=[];lineage=[];audits=[];optimizations=[]
    try:
        with h5py.File(DATASET) as handle:
            for name in sorted(handle['data'],key=lambda x:int(x.split('_')[-1])):
                number=int(name.split('_')[-1])
                if number>=40 or name in used:continue
                group=handle['data'][name]
                if float(group['rewards'][-1])<=0:continue
                f=_features(group)
                alignment=float(f['direction']@direction)
                # A new contact side and sufficient successful source motion
                # respond to the failed target's remaining motion direction.
                rear=float(-f['approach']@direction)
                score=2*alignment+3*f['displacement']+2*rear+.002*f['contact_window_steps']-.001*f['steps']
                candidates.append(dict(demo_id=name,score=score,alignment=alignment,
                    source_plate_displacement_m=f['displacement'],rear_contact_offset_m=rear,
                    contact_window=[f['lo'],f['hi']],initial_plate_xyz=f['initial_plate_xyz']))
            candidates.sort(key=lambda x:(-x['score'],x['demo_id']))
            selected=[]
            # Reject unreachable candidates before committing to selection.
            for candidate in candidates:
                group=handle['data'][candidate['demo_id']];f=_features(group)
                data.qpos[:]=default;data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes]=0
                best,trials=_optimize_offset(env,group,f,round_index,direction,contact_fraction,summary)
                candidate['optimization']=best
                if best['max_ik_error_m']>=.015 or best['max_orientation_error_rad']>=.15 or best['arm_collisions']:continue
                selected.append(candidate);optimizations.append(dict(demo_id=candidate['demo_id'],trials=trials,chosen=best,
                    baseline_objective=trials[0]['static_objective'],optimized_objective=best['static_objective'],
                    physical_dynamics_verified=False))
                if len(selected)>=2:break
            if not selected:raise RuntimeError('No feedback-selected source episode passed static feasibility')
            for candidate in selected:
                name=candidate['demo_id'];group=handle['data'][name];f=_features(group);sm,sd=source_model(group)
                offset=np.asarray(candidate['optimization']['offset_m']);n=f['steps']
                data.qpos[:]=default;data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes]=0
                poses=[];rotations=[];valid=[];qstates=[];observations=[]
                for index in range(n):
                    p,r=dataset_pose(group,sm,sd,index);copy_scene(sm,sd,model,data)
                    desired=p+offset*_blend(index,f['lo'],f['hi'])
                    result=ik(env,desired,r,max_iterations=30);collision=unintended_contacts(env)
                    feasible=bool(result['feasible'] and not collision)
                    poses.append(data.site_xpos[env.robots[0].eef_site_id].copy())
                    rotations.append(data.site_xmat[env.robots[0].eef_site_id].reshape(3,3).copy())
                    valid.append(feasible);qstates.append(canonical_state(env))
                    if index%2==0 and feasible:
                        observations.append((index,np.stack([env.sim.render(width=256,height=256,camera_name=c) for c in ('agentview','robot0_eye_in_hand')])))
                    audits.append(dict(source_demo_id=name,source_frame=index,**result,arm_collision_count=collision,
                                       correction_weight=_blend(index,f['lo'],f['hi'])))
                poses=np.asarray(poses);rotations=np.asarray(rotations);valid=np.asarray(valid)
                actions=np.zeros((n,7),np.float32)
                for index in range(n):
                    future=min(index+lookahead,n-1)
                    actions[index,:3]=np.clip((poses[future]-poses[index])/.05,-1,1)
                    actions[index,3:6]=np.clip(Rotation.from_matrix(rotations[future]@rotations[index].T).as_rotvec()/.5,-1,1)
                    actions[index,6]=float(group['actions'][index,6])
                action_valid=valid & valid[np.minimum(np.arange(n)+lookahead,n-1)]
                for index,pair in observations:
                    # Concentrate new supervision on the problematic contact and
                    # transport segment, while retaining original data globally.
                    if index<max(0,f['lo']-12) or index>min(n-1,f['hi']+10):continue
                    ids=np.arange(index,index+50);clipped=np.minimum(ids,n-1)
                    mask=(ids>=n)|~action_valid[clipped]
                    if np.all(mask):continue
                    images.append(pair);states.append(qstates[index]);chunks.append(actions[clipped]);masks.append(mask)
                    lineage.append(dict(source_demo_id=name,source_frame=index,origin='feedback_source_kinematic_retarget',
                                        executed_target_demonstration=False,round_index=round_index))
                progress('feedback_processing',dict(round_index=round_index,demo_id=name,new_anchors=len(images),offset_m=offset.tolist()))
            if not images:raise RuntimeError('No corrective source anchors survived')
            start=len(old_states)
            np.savez_compressed(out/'processed_source_data.npz',images=np.concatenate([old_images,np.stack(images)]),
                states=np.concatenate([old_states,np.stack(states)]),actions=np.concatenate([old_actions,np.stack(chunks)]),
                actions_id_pad=np.concatenate([old_mask,np.stack(masks)]))
            chosen=[x['demo_id'] for x in selected]
            selection={**old_selection,'selected':list(dict.fromkeys(old_selection.get('selected',[])+chosen)),
                'feedback_selected':list(dict.fromkeys(old_selection.get('feedback_selected',[])+chosen)),
                'feedback_round':round_index,'new_rows_start':start,'new_rows_count':len(images),
                'training_source_candidates_exclude_ids':list(range(40,50)),
                'feedback_method':'remaining_goal_direction_and_source_contact_geometry_ranking_then_static_distance_optimization',
                'diagnostic_used':summary,'target_execution_used_as_diagnosis':True,
                'executed_target_supervised_action_labels':False}
            write_json(out/'selection.json',selection)
            metadata=dict(round_index=round_index,parent_training_rows=start,new_rows_start=start,new_rows_count=len(images),
                total_training_rows=start+len(images),selected_new_source_ids=chosen,diagnostic_used=summary,
                direction_used=direction.tolist(),controller_lookahead_frames=lookahead,optimizations=optimizations,
                full_action_chunk_recomputed=True,actions_id_pad_included=True,
                target_physics_steps_during_processing=0,executed_target_training_demonstrations=0,
                pseudo_label_limit='static contact-distance improvement is not a verified target force/friction model',
                duration_seconds=time.monotonic()-started)
            write_json(out/'processing_audit.json',dict(**metadata,lineage=lineage,frames=audits))
            write_json(out/'feedback_processing.json',metadata)
            return metadata
    finally:
        env.close()
