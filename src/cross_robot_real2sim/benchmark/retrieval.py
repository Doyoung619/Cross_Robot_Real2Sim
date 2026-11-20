"""Frozen-VLA visual retrieval and two explicitly documented S-DTW variants."""
import json
from pathlib import Path
import numpy as np

from .core import dump


def subsequence_dtw(query,candidate,slope=False):
    q=np.asarray(query,float);c=np.asarray(candidate,float)
    if not len(q) or not len(c):return None
    distances=np.linalg.norm(q[:,None]-c[None,:],axis=-1);m,n=distances.shape
    if slope:
        # FERT Eq7: each query frame advances one or two candidate frames.
        d=np.full((m,n),np.inf);parents=np.full((m,n),-1,int);d[0]=distances[0]
        for i in range(1,m):
            for j in range(n):
                options=[(d[i-1,j-k],j-k) for k in [1,2] if j-k>=0]
                if options:
                    value,prev=min(options);d[i,j]=distances[i,j]+value;parents[i,j]=prev
        end=int(np.argmin(d[-1]));cost=float(d[-1,end]/m)
        if not np.isfinite(cost):return None
        j=end;indices=[j]
        for i in range(m-1,0,-1):j=int(parents[i,j]);indices.append(j)
        return dict(cost=cost,start=min(indices),end=end+1,path=list(reversed(indices)))
    d=np.full((m+1,n+1),np.inf);d[0,:]=0;parents={}
    for i in range(1,m+1):
        for j in range(1,n+1):
            value,prev=min([(d[i-1,j-1],(i-1,j-1)),(d[i-1,j],(i-1,j)),(d[i,j-1],(i,j-1))])
            d[i,j]=distances[i-1,j-1]+value;parents[i,j]=prev
    j=int(np.argmin(d[m,1:]))+1;end=j;i=m;path=[]
    while i>0:
        path.append(j-1);i,j=parents[i,j]
    return dict(cost=float(d[m,end]/max(len(path),1)),start=min(path),end=max(path)+1,path=list(reversed(path)))


class Encoder:
    """Frozen SmolVLA vision tower, coarse4x4 pool, fixed64D projection +8D state.

    This replaces STRAP's DINOv2/CLIP and adapts FERT's representation dimensions.
    It never fits statistics on evaluation observations.
    """
    def __init__(self,model,contract):self.model=model;self.contract=contract;self.projection=None
    def __call__(self,images,state):
        import torch
        import torch.nn.functional as F
        batch=self.contract.batch(images[None],state[None]);prepared,_=self.model.prepare_images(batch);parts=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for image in prepared:
                tokens=self.model.model.vlm_with_expert.embed_image(image).float();length=tokens.shape[1];side=int(length**.5)
                if side*side==length:
                    pooled=F.adaptive_avg_pool2d(tokens.transpose(1,2).reshape(1,-1,side,side),(4,4)).reshape(-1)
                else:pooled=tokens.mean(1).reshape(-1)
                parts.append(pooled)
        vector=torch.cat(parts).cpu().numpy()
        if self.projection is None:self.projection=np.random.default_rng(314159).normal(size=(len(vector),64)).astype(np.float32)/np.sqrt(len(vector))
        visual=vector@self.projection;visual/=max(np.linalg.norm(visual),1e-8)
        state=np.asarray(state,np.float32);mean=self.contract.mean['observation.state'].float().cpu().numpy();std=self.contract.std['observation.state'].float().cpu().numpy()
        state=(state-mean)/std;state/=max(np.linalg.norm(state),1e-8)
        return np.r_[visual,state].astype(np.float32)


def build_index(pool,model,contract,stride):
    path=pool.run/'shared/source_embeddings.npz';meta_path=pool.run/'shared/source_embeddings.json'
    encoder=Encoder(model,contract)
    if path.exists():
        z=np.load(path);meta=json.loads(meta_path.read_text());return {n:dict(features=z[n],frames=z[n+'_frames']) for n in meta['source_ids']},encoder
    values={};index={}
    for name in pool.names:
        group=pool.h['data'][name];frames=np.arange(0,len(group['actions']),stride);features=[]
        for t in frames:
            pair=np.stack([group['obs/agentview_rgb'][t],group['obs/eye_in_hand_rgb'][t]])
            state=np.r_[group['obs/ee_pos'][t],group['obs/ee_ori'][t],group['obs/gripper_states'][t]].astype(np.float32)
            features.append(encoder(pair,state))
        values[name]=np.stack(features);values[name+'_frames']=frames
        index[name]=dict(features=values[name],frames=frames)
    np.savez_compressed(path,**values)
    dump(meta_path,dict(source_ids=pool.names,stride=stride,representation='Frozen pretrained SmolVLA visual tower; coarse4x4 pooling when token grid is square; fixed64D Gaussian projection; separately normalized8D TCP/gripper state',
         evaluation_data_used=False,dimension=72))
    return index,encoder


def retrieve(index,feature_rows,horizon,count,failure=False):
    if failure:
        endpoint=horizon-10;begin=max(0,endpoint-90)
        rows=[r for r in feature_rows if begin<=r['step']<endpoint]
    else:rows=feature_rows
    if not rows:return [],dict(reason='empty_query',rankings=[])
    query=np.asarray([r['feature'] for r in rows]);ranked=[]
    for name,value in index.items():
        match=subsequence_dtw(query,value['features'],slope=failure)
        if match is None:continue
        frames=value['frames'];lo=int(frames[match['start']]);hi=int(frames[match['end']-1])+1
        ranked.append(dict(demo_id=name,start=max(0,lo-30),end=hi+30,cost=match['cost'],matched_start=lo,matched_end=hi))
    ranked.sort(key=lambda r:(r['cost'],r['demo_id']))
    return ranked[:count],dict(query_control_steps=[r['step'] for r in rows],rankings=ranked,
        slope_constraint='advance1or2' if failure else 'standard_three_way',
        target_success_demonstrations_used=0,replay_source='successful A demonstrations converted offline; not a native successful B buffer')
