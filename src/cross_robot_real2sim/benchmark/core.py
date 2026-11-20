"""Pure protocol contracts. This module never imports a simulator or policy."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import uuid


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def wilson(successes,n,z=1.959963984540054):
    if n<=0:return None,None
    if not 0<=successes<=n:raise ValueError('Invalid success count')
    p=successes/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (0. if successes==0 else max(0.,c-d)),(1. if successes==n else min(1.,c+d))


@dataclass(frozen=True)
class Method:
    id:str
    label:str
    selection:str='feasible'
    retarget:str='legacy'
    feedback:str='none'
    train_more:bool=True
    available:bool=True
    fidelity:str='ablation'
    unavailable_reason:str=''


METHODS={m.id:m for m in [
    Method('pretrained','Pretrained SmolVLA',selection='none',train_more=False),
    Method('random_lora','Random source + LoRA',selection='random'),
    Method('b_feasible_lora','B-feasible source + LoRA (frozen after K=0)',train_more=False),
    Method('fixed_source_train','Fixed-source additional LoRA'),
    Method('selection_only','Failure-guided selection only',feedback='selection'),
    Method('retarget_only','Failure-guided retargeting only',feedback='retarget'),
    Method('our_full','Failure-guided selection + retargeting',feedback='full'),
    Method('cei_inspired','CEI-inspired geometry alignment',retarget='functional',fidelity='inspired'),
    Method('strap_style','STRAP-style subtrajectory retrieval',selection='strap',feedback='strap',fidelity='style'),
    Method('fert_inspired','FERT-inspired failure retrieval',feedback='fert',fidelity='inspired'),
    Method('cei_official','CEI official',available=False,fidelity='not_reproduced',unavailable_reason='Full directional functional representations, global trajectory optimization and original observation/policy pipeline are not integrated.'),
    Method('strap_official','STRAP official',available=False,fidelity='not_reproduced',unavailable_reason='Original successful target demonstrations and original DINOv2/CLIP plus policy-learning protocol are not supplied in the common A-only supervision track.'),
    Method('fert_official','FERT official',available=False,fidelity='not_reproduced',unavailable_reason='Original successful target/fleet experience buffer, policy initialization and original fleet protocol are absent. No uncounted target teachers are generated.'),
]}


def validate(config):
    c=dict(config)
    if not c.get('methods') or any(m not in METHODS for m in c['methods']):raise ValueError('Unknown/empty method list')
    b=c['budgets']
    if b!=sorted(set(b)) or not b or b[0]!=0 or any(not isinstance(k,int) or k<0 for k in b):raise ValueError('Budgets must be sorted unique integers starting at0')
    if c['evaluation_states']<1 or c['horizon']<1:raise ValueError('Invalid evaluation size/horizon')
    if c['development_rollouts']!=0:raise ValueError('This version uses fixed schedules; development budget must be explicitly zero')
    if c['updates_initial']<1 or c['updates_per_rollout']<1:raise ValueError('Invalid training budget')
    if c['adaptation_states']<max(b):raise ValueError('Insufficient distinct adaptation states for the requested budget')
    if len(set(c['training_seeds']))!=len(c['training_seeds']):raise ValueError('Duplicate training seeds')
    return c


class Ledger:
    """A physical episode counts once, immediately before its first env.step.

    Static bank validation/IK/rendering never enters this ledger. Started but
    interrupted physical episodes remain counted. Evaluation cannot be handed
    to an adaptation method through a feedback envelope.
    """
    def __init__(self,path):self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def events(self):
        if not self.path.exists():return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x]
    def append(self,event):
        with self.path.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(event,allow_nan=False)+'\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)
    def begin(self,role,method,seed,state_id,checkpoint,budget_cap=None):
        if role not in ('adaptation','development','evaluation','smoke'):raise ValueError('Invalid rollout role')
        with self.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
            old=[json.loads(x) for x in f if x.strip()]
            if role=='adaptation' and budget_cap is not None:
                used=sum(e['event']=='started' and e['role']==role and e['method']==method and e['training_seed']==seed for e in old)
                if used>=budget_cap:raise RuntimeError('Target adaptation budget exhausted')
            episode=uuid.uuid4().hex
            event=dict(event='started',episode_id=episode,role=role,method=method,training_seed=seed,state_id=state_id,
                       checkpoint=checkpoint,at=datetime.now(timezone.utc).isoformat())
            f.seek(0,2);f.write(json.dumps(event)+'\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)
        return episode
    def end(self,episode_id,**details):self.append(dict(event='finished',episode_id=episode_id,**details))
    def count(self,role,method=None,seed=None):
        return sum(e['event']=='started' and e['role']==role and (method is None or e['method']==method) and
                   (seed is None or e['training_seed']==seed) for e in self.events())


def feedback_envelope(row,trace):
    if row['role']!='adaptation':raise ValueError('Only counted adaptation data may influence policy updates')
    return dict(role='adaptation',episode_id=row['episode_id'],metrics=row,trace=trace)


def require_feedback(value):
    if value.get('role')!='adaptation':raise ValueError('Evaluation/development data cannot enter this adaptation protocol')
    return value


def bootstrap():
    import sys
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root/'scripts'))
    from vla_pilot_common import bootstrap as setup
    setup()
    return root
