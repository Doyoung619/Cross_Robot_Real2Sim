"""Source-only supervision: retrieval, geometry alignment, static generation."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .core import dump,digest,object_hash,require_feedback


def source_features(group):
    from vla_feedback_processing import _features
    return _features(group)


def dcd(points_a,normals_a,points_b,normals_b,weight=.01):
    """Symmetric directional Chamfer (CEI Eq1), sparse functional patches."""
    matrix=np.linalg.norm(points_a[:,None]-points_b[None,:],axis=-1)-weight*(normals_a@normals_b.T)
    return float(matrix.min(axis=1).mean()+matrix.min(axis=0).mean())


def functional_patch(model,data,geoms):
    """Manually specified box-pad contact patches, not full mesh sampling."""
    if len(geoms)!=2:raise ValueError('Two parallel finger pads required')
    centers=data.geom_xpos[geoms];direction=centers[1]-centers[0];direction/=max(np.linalg.norm(direction),1e-9)
    points=[];normals=[]
    for number,g in enumerate(geoms):
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:raise ValueError('CEI-inspired adapter requires box finger pads')
        n=direction if number==0 else -direction;r=data.geom_xmat[g].reshape(3,3);local=r.T@n
        axis=int(np.argmax(np.abs(local)));normal=r[:,axis]*np.sign(local[axis]);size=model.geom_size[g]
        surface=centers[number]+normal*size[axis];tangent=[j for j in range(3) if j!=axis]
        for u,v in [(0,0),(-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)]:
            points.append(surface+u*size[tangent[0]]*r[:,tangent[0]]+v*size[tangent[1]]*r[:,tangent[1]])
            normals.append(normal)
    return np.asarray(points),np.asarray(normals)


def source_pad_ids(model):
    ids=[]
    for i in range(model.ngeom):
        name=model.geom(i).name or ''
        if 'finger' in name and 'pad' in name and (model.geom_contype[i] or model.geom_conaffinity[i]):ids.append(i)
    if len(ids)!=2:raise ValueError(f'Source pad annotation unavailable: {ids}')
    return ids


class SourcePool:
    def __init__(self,run):
        from vla_pilot_common import DATASET
        from .environment import configure,apply_calibration
        self.run=Path(run);self.dataset=DATASET;self.h=h5py.File(DATASET)
        self.env=configure(run).make_env(size=256)
        self.cal=np.load(self.run/'shared/camera_calibration.npz');apply_calibration(self.env,self.cal)
        from vla_pilot_processing import raw_target
        self.model,self.data=raw_target(self.env);self.default=self.data.qpos.copy()
        self.names=sorted(self.h['data'],key=lambda s:int(s.split('_')[-1]))
        assert len(self.names)==50 and all(float(self.h['data'][n]['rewards'][-1])>0 for n in self.names)
        self.metadata=json.loads((self.run/'shared/source_pool.json').read_text()) if (self.run/'shared/source_pool.json').exists() else None
    def close(self):self.env.close();self.h.close()
    def inspect(self):
        from vla_pilot_processing import source_model,dataset_pose,copy_scene,ik,unintended_contacts
        if self.metadata:return self.metadata
        start=time.monotonic();rows=[]
        for name in self.names:
            group=self.h['data'][name];sm,sd=source_model(group);self.data.qpos[:]=self.default;probes=[]
            for t in np.linspace(0,len(group['actions'])-1,4,dtype=int):
                p,r=dataset_pose(group,sm,sd,int(t));copy_scene(sm,sd,self.model,self.data)
                value=ik(self.env,p,r,max_iterations=160);value['arm_collision_count']=unintended_contacts(self.env);probes.append(value)
            f=source_features(group)
            rows.append(dict(demo_id=name,steps=f['steps'],initial_plate_xyz=f['initial_plate_xyz'],
                feasible=all(v['feasible'] and not v['arm_collision_count'] for v in probes),probes=probes))
        self.metadata=dict(episodes=rows,successful_source_episodes=50,eligible_count=sum(x['feasible'] for x in rows),
            wall_seconds=time.monotonic()-start,source_hdf5_sha256=digest(self.dataset),target_physics_steps=0,
            inherited_limits='15mm TCP position,0.15rad orientation, native joint limits, static robot-environment penetration test',
            note='All50 A demonstrations are the common source pool; fresh evaluation states are not HDF5 scene replays')
        dump(self.run/'shared/source_pool.json',self.metadata);return self.metadata
    def initial(self,kind,seed,count):
        rows=self.inspect()['episodes']
        if kind=='random':
            rng=np.random.default_rng(seed+12345);names=rng.choice(self.names,size=min(count,len(self.names)),replace=False).tolist()
        else:
            eligible=sorted([x for x in rows if x['feasible']],key=lambda x:(x['steps'],x['demo_id']))
            if not eligible:return []
            chosen=[eligible.pop(0)]
            while eligible and len(chosen)<count:
                b=max(eligible,key=lambda x:min(np.linalg.norm(np.array(x['initial_plate_xyz'])-y['initial_plate_xyz']) for y in chosen))
                chosen.append(b);eligible.remove(b)
            names=[x['demo_id'] for x in chosen]
        return [dict(demo_id=n,start=0,end=len(self.h['data'][n]['actions'])) for n in names]
    def retrieve_geometry(self,diagnostic,used,count):
        direction=np.asarray(diagnostic['mean_remaining_goal_vector_xy']);direction/=max(np.linalg.norm(direction),1e-9)
        ranked=[]
        for name in self.names:
            if name in used:continue
            f=source_features(self.h['data'][name]);alignment=float(f['direction']@direction);rear=float(-f['approach']@direction)
            value=2*alignment+3*f['displacement']+2*rear+.002*f['contact_window_steps']-.001*f['steps']
            ranked.append(dict(demo_id=name,score=value,start=max(0,f['lo']-12),end=min(f['steps'],f['hi']+11)))
        ranked.sort(key=lambda r:(-r['score'],r['demo_id']))
        return ranked
    def offset(self,group,diagnostic,index):
        """Reuse bounded static objective; force the same160-iteration IK cap."""
        from vla_feedback_processing import _optimize_offset
        import vla_feedback_processing as fp
        from vla_pilot_processing import ik
        native=fp.ik
        def humanoid_ik(env,p,r,max_iterations=50):return ik(env,p,r,max_iterations=max(160,max_iterations))
        fp.ik=humanoid_ik
        try:
            f=source_features(group);direction=np.asarray(diagnostic['mean_remaining_goal_vector_xy']);direction/=max(np.linalg.norm(direction),1e-9)
            self.data.qpos[:]=self.default
            best,trials=_optimize_offset(self.env,group,f,index,direction,diagnostic['contact_fraction'],diagnostic)
            okay=best['max_ik_error_m']<.015 and best['max_orientation_error_rad']<.15 and not best['arm_collisions']
            return np.asarray(best['offset_m']),dict(chosen=best,trials=trials,feasible=bool(okay))
        finally:fp.ik=native
    def generate(self,specs,mode='legacy',diagnostic=None,round_index=0,feedback_episode=None):
        """No physics: source restore + target forward kinematics/render only."""
        from vla_pilot_processing import source_model,dataset_pose,copy_scene,ik,canonical_state,unintended_contacts
        from vla_feedback_processing import _blend
        key=object_hash(dict(specs=specs,mode=mode,diagnostic=diagnostic,round_index=round_index,
                             camera=digest(self.run/'shared/camera_calibration.npz')))
        folder=self.run/'shared/data'/key;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'data.npz').exists():
            value=json.loads((folder/'audit.json').read_text());value['cache_hit']=True
            return folder/'data.npz',value
        start=time.monotonic();images=[];states=[];chunks=[];masks=[];lineage=[];frames=[];selection=[];rejections=[]
        for spec in specs:
            name=spec['demo_id'];group=self.h['data'][name];f=source_features(group);offset=np.zeros(3);optimization=None
            if mode=='corrected':
                offset,optimization=self.offset(group,diagnostic,round_index)
                if not optimization['feasible']:rejections.append(dict(**spec,reason='static_contact_offset_infeasible',optimization=optimization));continue
            sm,sd=source_model(group);n=len(group['actions']);self.data.qpos[:]=self.default
            poses=[];rotations=[];valid=[];per_state=[];observations=[];last_q=self.default[self.env.robots[0]._ref_joint_pos_indexes].copy()
            source_pads=source_pad_ids(sm) if mode=='functional' else None
            pads=[self.model.geom(self.env.robots[0].gripper.important_geoms[k][0]).id for k in ['left_fingerpad','right_fingerpad']]
            for t in range(n):
                p,r=dataset_pose(group,sm,sd,t);copy_scene(sm,sd,self.model,self.data)
                goal=p+offset*_blend(t,f['lo'],f['hi']);result=ik(self.env,goal,r,max_iterations=160)
                functional=None
                if mode=='functional' and result['feasible'] and t%2==0:
                    pa,na=functional_patch(sm,sd,source_pads);qids=np.asarray(self.env.robots[0]._ref_joint_pos_indexes)
                    lo,hi=self.model.jnt_range[np.asarray(self.env.robots[0]._ref_joint_indexes)].T
                    def cost(q):
                        self.data.qpos[qids]=q;mujoco.mj_forward(self.model,self.data)
                        pb,nb=functional_patch(self.model,self.data,pads)
                        tcp=self.data.site_xpos[self.env.robots[0].eef_site_id]
                        return dcd(pa,na,pb,nb)+2*np.linalg.norm(tcp-goal)+.001*np.sum((q-last_q)**2)
                    q0=self.data.qpos[qids].copy();before=cost(q0)
                    res=minimize(cost,q0,method='L-BFGS-B',bounds=list(zip(lo+.002,hi-.002)),options={'maxiter':12,'ftol':1e-7})
                    q=res.x if np.isfinite(res.fun) and res.fun<=before else q0;after=cost(q)
                    self.data.qpos[qids]=q;mujoco.mj_forward(self.model,self.data)
                    pe=float(np.linalg.norm(goal-self.data.site_xpos[self.env.robots[0].eef_site_id]));re=float(np.linalg.norm(Rotation.from_matrix(r@self.data.site_xmat[self.env.robots[0].eef_site_id].reshape(3,3).T).as_rotvec()))
                    result.update(position_error_m=pe,orientation_error_rad=re,feasible=pe<.015 and re<.15)
                    functional=dict(before=before,after=after,optimizer='bounded L-BFGS-B numerical gradients',points_per_pad=5)
                collision=unintended_contacts(self.env);okay=bool(result['feasible'] and not collision)
                poses.append(self.data.site_xpos[self.env.robots[0].eef_site_id].copy());rotations.append(self.data.site_xmat[self.env.robots[0].eef_site_id].reshape(3,3).copy())
                valid.append(okay);per_state.append(canonical_state(self.env))
                frames.append(dict(demo_id=name,frame=t,**result,arm_collision_count=collision,functional=functional))
                if t%2==0 and spec['start']<=t<spec['end'] and okay:
                    observations.append((t,np.stack([self.env.sim.render(width=256,height=256,camera_name=c) for c in ['agentview','robot0_eye_in_hand']])))
                last_q=self.data.qpos[self.env.robots[0]._ref_joint_pos_indexes].copy()
            poses=np.asarray(poses);rotations=np.asarray(rotations);valid=np.asarray(valid)
            if mode=='corrected':
                gain=(diagnostic or {}).get('response_gain')
                gain=float(gain) if gain is not None and np.isfinite(gain) and gain>0 else .25
                lookahead=int(np.clip(round(1/gain),2,5))
            else:lookahead=1 if mode=='functional' else 0
            labels=np.asarray(group['actions'],dtype=np.float32).copy()
            if lookahead:
                for t in range(n):
                    j=min(t+lookahead,n-1);labels[t,:3]=np.clip((poses[j]-poses[t])/.05,-1,1)
                    labels[t,3:6]=np.clip(Rotation.from_matrix(rotations[j]@rotations[t].T).as_rotvec()/.5,-1,1)
            action_valid=valid & valid[np.minimum(np.arange(n)+lookahead,n-1)]
            before_count=len(images)
            for t,pair in observations:
                idx=np.arange(t,t+50);clipped=np.minimum(idx,n-1);mask=(idx>=n)|~action_valid[clipped]
                if mask.all():continue
                label=labels[clipped].copy()
                if not lookahead:
                    source_p,_=dataset_pose(group,sm,sd,t);label[0,:3]=np.clip(label[0,:3]+(source_p-poses[t])/.05,-1,1)
                images.append(pair);states.append(per_state[t]);chunks.append(label);masks.append(mask)
                lineage.append(dict(source_demo_id=name,source_frame=t,mode=mode,feedback_episode_id=feedback_episode,supervised_target_execution=False))
            selection.append(dict(**spec,accepted_anchors=len(images)-before_count,offset_m=offset.tolist(),optimization=optimization))
        audit=dict(mode=mode,requested=specs,selected=selection,rejected=rejections,anchors=len(images),lineage=lineage,frames=frames,
            cache_hit=False,
            physical_target_steps=0,target_action_demonstrations=0,wall_seconds=time.monotonic()-start,feedback_episode_id=feedback_episode,
            label_limit='Static geometry with heuristic Cartesian command labels; no verified target contact dynamics',
            common_mask='All methods mask out-of-episode and statically infeasible current/future labels; stricter than the legacy unmasked initial pilot')
        dump(folder/'audit.json',audit)
        if len(images)<20:
            audit['unavailable_reason']=f'Only {len(images)} anchors; common minimum20 not met';dump(folder/'audit.json',audit);return None,audit
        np.savez_compressed(folder/'data.npz',images=np.stack(images),states=np.stack(states),actions=np.stack(chunks),actions_id_pad=np.stack(masks))
        return folder/'data.npz',audit


def combine(old,new,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with np.load(old) as a,np.load(new) as b:
        start=len(a['states']);np.savez_compressed(out,**{k:np.concatenate([a[k],b[k]]) for k in a.files})
    return out,start
