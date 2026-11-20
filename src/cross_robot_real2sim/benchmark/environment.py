"""Static state banks and a single instrumented physical-rollout entry point."""
from __future__ import annotations
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from .core import dump,digest,object_hash


def configure(run):
    from . import gr1
    gr1.GENERATED=Path(run)/'shared/gr1_model'
    return gr1


def calibrate(env,path):
    import h5py
    from vla_pilot_common import DATASET
    from vla_pilot_processing import source_model,dataset_pose,copy_scene,ik,raw_target,camera_calibration,canonical_state
    m,d=raw_target(env)
    with h5py.File(DATASET) as h:
        g=h['data/demo_5'];sm,sd=source_model(g);p,r=dataset_pose(g,sm,sd,0);copy_scene(sm,sd,m,d)
        d.qpos[env.robots[0]._ref_gripper_joint_pos_indexes]=0
        result=ik(env,p,r,max_iterations=160)
        if not result['feasible']:raise RuntimeError('Fixed shared initial GR1 TCP is infeasible; constraints are not relaxed')
        camera_calibration(env,sm,sd)
    values=dict(positions=m.cam_pos.copy(),quaternions=m.cam_quat.copy(),fovy=m.cam_fovy.copy(),
                state_rotation_offset=env._pilot_state_rotation_offset.copy(),
                robot_qpos=d.qpos[env.robots[0]._ref_joint_pos_indexes].copy(),
                gripper_qpos=d.qpos[env.robots[0]._ref_gripper_joint_pos_indexes].copy())
    np.savez_compressed(path,**values)
    return values


def apply_calibration(env,cal):
    from vla_pilot_processing import raw_target
    import mujoco
    m,d=raw_target(env)
    m.cam_pos[:]=cal['positions'];m.cam_quat[:]=cal['quaternions'];m.cam_fovy[:]=cal['fovy']
    env._pilot_state_rotation_offset=cal['state_rotation_offset'].copy()
    d.qpos[env.robots[0]._ref_joint_pos_indexes]=cal['robot_qpos']
    d.qpos[env.robots[0]._ref_gripper_joint_pos_indexes]=cal['gripper_qpos']
    d.qvel[:]=0;d.qacc[:]=0;d.time=0
    mujoco.mj_forward(m,d)


def reset_controller(env):
    r=env.robots[0];r.controller.update(force=True)
    r.controller.update_initial_joints(env.sim.data.qpos[r._ref_joint_pos_indexes].copy())
    r.controller.reset_goal();r.gripper.current_action=np.array([-1.])


def make_banks(run,config):
    """Native BDDL placement distributions, no physics settling or reward search."""
    import mujoco
    from vla_pilot_processing import raw_target,unintended_contacts,canonical_state
    run=Path(run);folder=run/'states';folder.mkdir(parents=True,exist_ok=True)
    gr1=configure(run);env=gr1.make_env(size=256)
    try:
        cal=calibrate(env,run/'shared/camera_calibration.npz')
        all_signatures=set();all_plate=[];bank_meta={}
        for role,number,offset in [('adaptation',config['adaptation_states'],0),('evaluation',config['evaluation_states'],100000),('development',0,200000)]:
            rows=[];arrays={k:[] for k in ['state','body_pos','body_quat','site_pos']};rejections=[];attempt=0
            while len(rows)<number and attempt<max(number*20,20):
                seed=config['state_seed']+offset+attempt;attempt+=1
                np.random.seed(seed);env.seed(seed);env.reset();apply_calibration(env,cal)
                m,d=raw_target(env);plate=m.body('plate_1_main').id
                xy=d.xpos[plate,:2].copy();state=env.get_sim_state().copy()
                values=dict(state=state,body_pos=m.body_pos.copy(),body_quat=m.body_quat.copy(),site_pos=m.site_pos.copy())
                signature=object_hash({k:v.tolist() for k,v in values.items()})
                reason=None
                if signature in all_signatures:reason='duplicate_full_state'
                elif any(np.linalg.norm(xy-v)<1e-5 for v in all_plate):reason='plate_positions_not_distinct_at_10um'
                elif env.check_success():reason='already_successful'
                elif unintended_contacts(env):reason='static_robot_environment_penetration'
                elif not all(np.isfinite(v).all() for v in values.values()):reason='nonfinite_state'
                if reason:rejections.append(dict(seed=seed,reason=reason));continue
                all_signatures.add(signature);all_plate.append(xy)
                row=dict(id=f'{role}_{len(rows):04d}',role=role,simulator_seed=seed,index=len(rows),signature=signature,
                         plate_xy=xy.tolist(),initial_canonical_state=canonical_state(env).tolist(),validation='static_native_reset_valid',physics_steps=0)
                rows.append(row)
                for k,v in values.items():arrays[k].append(v)
            if len(rows)!=number:raise RuntimeError(f'Only {len(rows)}/{number} unique valid {role} starts; see state rejection log')
            if number:np.savez_compressed(folder/f'{role}.npz',**{k:np.stack(v) for k,v in arrays.items()})
            meta=dict(role=role,count=len(rows),states=rows,rejections=rejections,physics_steps_used_for_validation=0,
                      distribution='Native LIBERO BDDL sampler; plate x=[.04,.06], y=[-.03,-.01]; other objects/fixtures sampled natively',
                      note='Distinct narrow in-distribution initial states, not broad OOD generalization; no reuse of the old two scenes')
            dump(folder/f'{role}.json',meta);bank_meta[role]=meta
        dump(folder/'manifest.json',dict(counts={k:v['count'] for k,v in bank_meta.items()},
             all_full_states_disjoint=True,all_plate_positions_distinct=True,camera_sha256=digest(run/'shared/camera_calibration.npz'),
             files={f.name:digest(f) for f in sorted(folder.glob('*')) if f.name!='manifest.json'}))
        return bank_meta
    finally:env.close()


class Bank:
    def __init__(self,run,role):
        if role not in ('adaptation','evaluation','development'):raise ValueError(role)
        self.run=Path(run);self.role=role;self.meta=json.loads((self.run/f'states/{role}.json').read_text())
        self.data=np.load(self.run/f'states/{role}.npz') if self.meta['count'] else None
        self.cal=np.load(self.run/'shared/camera_calibration.npz')
    def __len__(self):return self.meta['count']
    def restore(self,env,index):
        from vla_pilot_processing import raw_target,canonical_state
        import mujoco
        row=self.meta['states'][index]
        env.reset();m,d=raw_target(env)
        for k in ['body_pos','body_quat','site_pos']:getattr(m,k)[:]=self.data[k][index]
        apply_calibration(env,self.cal)
        env.set_state(self.data['state'][index]);mujoco.mj_forward(m,d);reset_controller(env)
        if env.check_success():raise RuntimeError('Sealed start unexpectedly satisfies goal')
        np.testing.assert_array_equal(canonical_state(env),row['initial_canonical_state'])
        return row


def rollout(env,bank,index,policy,contract,ledger,method,seed,checkpoint,config,role,path,
            budget_cap=None,encoder=None,record=False):
    """All B env.step calls for benchmark learning/evaluation live here."""
    if bank.role!=role:raise ValueError('State-bank role mismatch')
    import torch
    from vla_pilot_processing import raw_target,canonical_state,unintended_contacts
    from vla_feedback_rollouts import _contact_ids,_contact_measurement,_aggregate
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    startrow=bank.restore(env,index);torch.manual_seed(918273+seed*10000+index)
    policy.reset();m,d=raw_target(env);r=env.robots[0];plate=m.body('plate_1_main').id;goal=m.site('main_table_stove_front_region').id
    initial_plate=d.xpos[plate].copy();initial_state=canonical_state(env).tolist();contact_ids=_contact_ids(env)
    qids=np.asarray(r._ref_joint_pos_indexes);lo,hi=m.jnt_range[np.asarray(r._ref_joint_indexes)].T
    queue=deque();trace=[];feature_rows=[];hold=0;ever=False;episode=None;error=None;collisions=0
    wall=time.monotonic();writer=None;physical_steps=0;inference_cuda_s=0.
    torch.cuda.reset_peak_memory_stats()
    if record:
        import imageio.v2 as imageio
        writer=imageio.get_writer(path/'rollout.mp4',fps=10,codec='libx264',quality=7,macro_block_size=1)
    try:
        for step in range(config['horizon']):
            if not queue:
                pair=np.stack([env.sim.render(width=256,height=256,camera_name=c) for c in ['agentview','robot0_eye_in_hand']])
                state=canonical_state(env);batch=contract.batch(pair[None],state[None])
                cuda_begin=torch.cuda.Event(enable_timing=True);cuda_end=torch.cuda.Event(enable_timing=True);cuda_begin.record()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):chunk=policy.predict_action_chunk(batch)
                decoded=contract.decode(chunk)[0]
                queue.extend(decoded[:5])
                if encoder is not None:feature_rows.append(dict(step=step,feature=encoder(pair,state).tolist()))
                cuda_end.record();cuda_end.synchronize();inference_cuda_s+=cuda_begin.elapsed_time(cuda_end)/1000
            raw_action=np.asarray(queue.popleft());action=np.clip(raw_action,-1,1)
            if not np.isfinite(action).all():raise RuntimeError('Nonfinite policy action')
            if episode is None:episode=ledger.begin(role,method,seed,startrow['id'],checkpoint,budget_cap)
            previous=d.xpos[plate].copy();physical_steps+=1;env.step(action)
            success=bool(env.check_success());ever|=success;hold=hold+1 if success else 0
            p=d.xpos[plate].copy();g=d.site_xpos[goal].copy();eef=canonical_state(env)
            direction=g[:2]-p[:2];den=np.linalg.norm(action[:2])*np.linalg.norm(direction)
            command=np.linalg.norm(action[:2])*.05
            margin=float(np.min(np.minimum(d.qpos[qids]-lo,hi-d.qpos[qids])))
            c=unintended_contacts(env);collisions+=c>0
            row=dict(step=step,time_s=(step+1)/20,eef_xyz=eef[:3].tolist(),plate_xyz=p.tolist(),goal_xyz=g.tolist(),
                sim_state=env.get_sim_state().tolist(),
                action=action.tolist(),goal_error_m=float(np.linalg.norm(direction)),native_success=success,
                push_direction_alignment=float(action[:2]@direction/den) if den>1e-6 else None,
                plate_response_gain=float(np.linalg.norm(p[:2]-previous[:2])/command) if command>1e-4 else None,
                eef_tracking_error_m=float(np.linalg.norm(r.controller.goal_pos-eef[:3])),
                joint_limit_margin_rad=margin,joint_limit_violation=margin<0,arm_collision_count=int(c),**_contact_measurement(env,contact_ids))
            trace.append(row)
            if writer and step%2==0:
                from .gr1 import frame
                writer.append_data(frame(env,method,startrow['id'],step,success,row['goal_error_m']))
            if hold>=config['success_hold_steps']:break
    except Exception as exc:error=repr(exc)
    finally:
        if writer:writer.close()
    aggregate=_aggregate(trace,initial_plate) if trace else {}
    result=dict(role=role,method=method,training_seed=seed,checkpoint=checkpoint,state_id=startrow['id'],state_signature=startrow['signature'],
        simulator_seed=startrow['simulator_seed'],episode_id=episode,physical_rollout_count=int(episode is not None),steps=len(trace),
        success=bool(hold>=config['success_hold_steps'] and error is None),ever_success=ever,stable_success_steps=hold,
        native_success_at_final=bool(env.check_success()),
        goal_error_m=float(np.linalg.norm(d.xpos[plate,:2]-d.site_xpos[goal,:2])),
        initial_plate_position=initial_plate.tolist(),initial_canonical_state=initial_state,
        plate_displacement_m=float(np.linalg.norm(d.xpos[plate]-initial_plate)),
        remaining_goal_vector_xy=(d.site_xpos[goal,:2]-d.xpos[plate,:2]).tolist(),arm_collision_steps=collisions,
        mean_eef_tracking_error_m=float(np.mean([x['eef_tracking_error_m'] for x in trace])) if trace else None,
        joint_limit_violation_steps=sum(x['joint_limit_violation'] for x in trace),
        minimum_joint_limit_margin_rad=min((x['joint_limit_margin_rad'] for x in trace),default=None),
        frozen_body_max_position_drift_m=env._max_fixed_position_drift,
        frozen_body_max_quaternion_drift=env._max_fixed_quaternion_drift,
        training_usage='diagnostic_only_no_B_action_targets' if role=='adaptation' else 'sealed_final_evaluation_never_feedback',
        wall_seconds=time.monotonic()-wall,attempted_physical_steps=physical_steps,
        inference_cuda_elapsed_seconds=inference_cuda_s,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
        error=error,**aggregate)
    if episode:ledger.end(episode,physical_steps=physical_steps,success=result['success'],error=error)
    dump(path/'result.json',result)
    dump(path/'trace.json',dict(role=role,steps=trace,query_features=feature_rows,actions_are_not_supervised_targets=True))
    return result,trace,feature_rows
