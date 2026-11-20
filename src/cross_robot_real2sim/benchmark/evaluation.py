"""Sealed final evaluation, invoked only after adaptation is complete."""
from pathlib import Path
import json
import time

from .core import dump,Ledger,object_hash
from .environment import Bank,configure,rollout


def evaluate(run,config,progress,deadline=None):
    import torch
    from vla_pilot_learning import load_model,InputContract
    run=Path(run)
    if not (run/'TRAINING_COMPLETE.json').exists():raise RuntimeError('Final evaluation is locked until every requested adaptation method is finished')
    if not (run/'EVALUATION_STARTED.json').exists():dump(run/'EVALUATION_STARTED.json',dict(started=True,training_closed=True))
    bank=Bank(run,'evaluation');ledger=Ledger(run/'rollouts.jsonl');env=configure(run).make_env(size=256)
    checkpoints=json.loads((run/'checkpoints.json').read_text());aliases=[]
    try:
        for c in checkpoints:
            key=object_hash(dict(checkpoint=c['checkpoint_sha256'],training_seed=c['training_seed'],
                states=bank.meta,horizon=config['horizon'],hold=config['success_hold_steps'],inference_seed=918273))
            path=run/'evaluations'/key;path.mkdir(parents=True,exist_ok=True)
            existing=sorted(path.glob('episode_*/result.json'))
            if len(existing)<len(bank):
                model=load_model(adapter=c['checkpoint'] is not None,adapter_path=run/c['checkpoint'] if c['checkpoint'] else None);contract=InputContract(model)
                try:
                    for i in range(len(bank)):
                        folder=path/f'episode_{i:04d}'
                        if (folder/'result.json').exists():continue
                        if deadline and time.time()>deadline-30:break
                        row,_,_=rollout(env,bank,i,model,contract,ledger,c['method'],c['training_seed'],c['checkpoint_sha256'],config,'evaluation',folder,
                            record=i<config['record_evaluation_videos'])
                        progress('evaluation_episode',dict(method=c['method'],budget_K=c['budget_K'],index=i,success=row['success'],wall_seconds=row['wall_seconds']))
                finally:del model,contract;torch.cuda.empty_cache()
            rows=[json.loads(f.read_text()) for f in sorted(path.glob('episode_*/result.json'))]
            for row in rows:
                if row['role']!='evaluation':raise RuntimeError('Non-evaluation row in sealed final cache')
            alias=dict(**c,evaluation_key=key,evaluation_trials=len(rows),requested_evaluation_trials=len(bank),
                evaluation_status='completed' if len(rows)==len(bank) else 'partial',
                evaluation_cache_note='Only byte-identical policy weights, same stochastic inference seed and same full state bank may share these trials; aliases are not extra independent observations')
            aliases.append(alias);dump(run/'evaluated_checkpoints.json',aliases)
        dump(run/'EVALUATION_FINISHED.json',dict(requested_checkpoints=len(checkpoints),recorded_checkpoints=len(aliases),
            actual_evaluation_rollouts=ledger.count('evaluation'),adaptation_rollouts=ledger.count('adaptation'),
            development_rollouts=ledger.count('development'),evaluation_used_for_learning=False))
    finally:env.close()
