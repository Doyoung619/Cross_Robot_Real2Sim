"""Adaptation phase only. Evaluation artifacts are never read here."""
from pathlib import Path
import json
import time

import numpy as np

from .core import METHODS,Ledger,dump,digest,feedback_envelope,require_feedback
from .data import SourcePool,combine
from .environment import Bank,rollout
from .learning import train


def checkpoint_record(run,folder,method,seed,budget,actual,checkpoint,data,updates,lineage,costs,reason=None):
    record=dict(method=method,training_seed=seed,budget_K=budget,actual_adaptation_rollouts=actual,
        development_rollouts=0,checkpoint=str(Path(checkpoint).relative_to(run)) if checkpoint else None,
        checkpoint_sha256=digest(checkpoint) if checkpoint else 'pretrained',data=str(Path(data).relative_to(run)) if data else None,
        optimizer_updates=updates,unique_source_demos=sorted({r['source_demo_id'] for r in lineage}),training_anchors=len(lineage),
        cost_events=list(costs),checkpoint_selection='last_predeclared_update_no_evaluation_model_selection',budget_shortfall_reason=reason)
    dump(folder/f'K_{budget:02d}.json',record);return record


def adapt(run,config,progress):
    import torch
    from vla_pilot_learning import load_model,InputContract
    from vla_feedback_rollouts import diagnose
    from .retrieval import build_index,Encoder,retrieve
    run=Path(run);ledger=Ledger(run/'rollouts.jsonl');bank=Bank(run,'adaptation');pool=SourcePool(run)
    if (run/'EVALUATION_STARTED.json').exists():raise RuntimeError('Adaptation is locked after final evaluation starts')
    records=[];status=[];index=None;encoding_cost=0.
    try:
        if any(METHODS[m].selection=='strap' or METHODS[m].feedback in ('strap','fert') for m in config['methods']):
            start=time.monotonic();model=load_model(False);contract=InputContract(model)
            index,encoder=build_index(pool,model,contract,config['feature_stride'])
            encoding_cost=time.monotonic()-start;del model,contract,encoder;torch.cuda.empty_cache()
            dump(run/'shared/embedding_cost.json',dict(wall_seconds=encoding_cost,attribution='shared preparation; reported separately, not silently charged only to the first retrieval method'))
        for seed in config['training_seeds']:
            for name in config['methods']:
                method=METHODS[name];folder=run/'methods'/f'seed_{seed}'/name;folder.mkdir(parents=True,exist_ok=True)
                if (folder/'done.json').exists():
                    records.extend(json.loads(f.read_text()) for f in sorted(folder.glob('K_*.json')));continue
                start=time.monotonic();progress('method_started',dict(method=name,seed=seed))
                if not method.available:
                    value=dict(method=name,seed=seed,status='unavailable',reason=method.unavailable_reason);status.append(value);dump(folder/'done.json',value);continue
                if name=='pretrained':
                    for k in config['budgets']:records.append(checkpoint_record(run,folder,name,seed,k,0,None,None,0,[],[],reason='Reference policy needs no adaptation interaction'))
                    dump(folder/'done.json',dict(status='completed',method=name,wall_seconds=time.monotonic()-start));continue
                specs=pool.initial(method.selection,seed,config['source_episodes']);query_retrieval=None
                if method.selection=='strap':
                    shared_specs=pool.initial('feasible',seed,config['source_episodes']);qdata,qaudit=pool.generate(shared_specs)
                    if qdata:
                        model=load_model(False);contract=InputContract(model);encoder=Encoder(model,contract);query=[]
                        with np.load(qdata) as z:
                            for j in range(0,len(z['states']),4):query.append(dict(step=j*2,feature=encoder(z['images'][j],z['states'][j]).tolist()))
                        specs,query_retrieval=retrieve(index,query,max((r['step'] for r in query),default=0)+1,config['source_episodes'])
                        del model,contract,encoder;torch.cuda.empty_cache()
                    else:specs=[]
                try:data,audit=pool.generate(specs,mode=method.retarget)
                except (ValueError,NotImplementedError) as exc:
                    value=dict(method=name,seed=seed,status='unavailable_integration',reason=repr(exc));dump(folder/'done.json',value);status.append(value);continue
                dump(folder/'initial_selection.json',dict(specs=specs,retarget=method.retarget,retrieval=query_retrieval,audit_path=str(pool.run/'shared/data')))
                if data is None:
                    value=dict(method=name,seed=seed,status='unavailable',reason=audit.get('unavailable_reason','No source data'),audit=audit);dump(folder/'done.json',value);status.append(value);continue
                lineage=list(audit['lineage']);costs=[dict(phase='initial_data',wall_seconds=0. if audit.get('cache_hit') else audit['wall_seconds'],cache_hit=audit.get('cache_hit',False),equivalent_uncached_wall_seconds=audit['wall_seconds'],cache_key=str(data.parent.name))]
                ckpt,learning,hit=train(run,data,None,config['updates_initial'],seed,0)
                costs.append(dict(phase='initial_training',cache_hit=hit,**learning));costs[-1]['wall_seconds']=0. if hit else learning['wall_seconds'];updates=config['updates_initial'];actual=0
                initial_specs=specs;priorities=np.ones(len(lineage));reason=None
                records.append(checkpoint_record(run,folder,name,seed,0,actual,ckpt,data,updates,lineage,costs))
                if not method.train_more:
                    for k in config['budgets'][1:]:records.append(checkpoint_record(run,folder,name,seed,k,0,ckpt,data,updates,lineage,costs,reason='Frozen source-adapted reference; no additional interactions or training requested'))
                    dump(folder/'done.json',dict(status='completed',method=name,wall_seconds=time.monotonic()-start));continue
                for k in range(1,max(config['budgets'])+1):
                    model=load_model(True,ckpt);contract=InputContract(model)
                    encoder=Encoder(model,contract) if method.feedback in ('strap','fert') else None
                    row,trace,features=rollout(pool.env,bank,k-1,model,contract,ledger,name,seed,str(ckpt),config,'adaptation',folder/f'adaptation_{k:02d}',
                        budget_cap=max(config['budgets']),encoder=encoder,record=(k==1))
                    del model,contract,encoder;torch.cuda.empty_cache()
                    actual=ledger.count('adaptation',name,seed);costs.append(dict(phase='adaptation_rollout',wall_seconds=row['wall_seconds'],episode_id=row['episode_id'],
                        inference_cuda_elapsed_seconds=row['inference_cuda_elapsed_seconds'],peak_torch_allocated_bytes=row['peak_torch_allocated_bytes']))
                    envelope=feedback_envelope(row,trace);require_feedback(envelope)
                    if row['episode_id'] is None or not trace:
                        reason='Policy/runtime failed before a usable physical adaptation episode';dump(folder/f'feedback_{k:02d}.json',dict(reason=reason,metrics=row));break
                    diagnostic=diagnose([row]);new_start=None;new_data=None;new_audit=None;selection=None
                    selection_log=dict(input_episode_id=row['episode_id'],diagnostic=diagnostic,feedback_mode=method.feedback,
                        target_actions_used_as_labels=False,source_pool='same50 successful Panda demonstrations')
                    update_start=time.monotonic()
                    if method.feedback in ('full','selection'):
                        ranked=pool.retrieve_geometry(diagnostic,{x['source_demo_id'] for x in lineage},config['new_source_episodes']);chosen=[]
                        for candidate in ranked:
                            if method.feedback=='full':
                                _,probe=pool.offset(pool.h['data'][candidate['demo_id']],diagnostic,k)
                                okay=probe['feasible']
                            else:
                                meta=next(x for x in pool.inspect()['episodes'] if x['demo_id']==candidate['demo_id']);okay=meta['feasible']
                            if okay:chosen.append(candidate)
                            if len(chosen)>=config['new_source_episodes']:break
                        selection=chosen;selection_log.update(rankings=ranked,chosen=chosen)
                        if chosen:new_data,new_audit=pool.generate(chosen,'corrected' if method.feedback=='full' else 'legacy',diagnostic,k,row['episode_id'])
                    elif method.feedback=='retarget':
                        selection=initial_specs;new_data,new_audit=pool.generate(selection,'corrected',diagnostic,k,row['episode_id']);selection_log['chosen']=selection
                    elif method.feedback in ('strap','fert'):
                        if method.feedback=='fert' and row['success']:
                            selection=[];selection_log['reason']='Successful adaptation rollout is not a failure query; no B actions are replayed'
                        else:
                            selection,retrieval=retrieve(index,features,len(trace),config['new_source_episodes'],failure=method.feedback=='fert');selection_log.update(chosen=selection,retrieval=retrieval)
                            if selection:new_data,new_audit=pool.generate(selection,'legacy',diagnostic,k,row['episode_id'])
                    if new_data is not None:
                        data,new_start=combine(data,new_data,folder/f'data_after_{k:02d}.npz');lineage+=new_audit['lineage']
                        priorities=np.r_[priorities,np.ones(len(new_audit['lineage']))]
                        selection_log.update(new_anchors=new_audit['anchors'],data_audit=str(new_data.parent/'audit.json'),
                            offsets=[dict(demo_id=r['demo_id'],offset_m=r['offset_m']) for r in new_audit['selected']])
                    elif method.feedback!='none':selection_log['no_new_data_reason']='No unused source passed static feasibility / common minimum20 anchors, or no valid retrieval. Constraints retained; matched additional training uses existing data.'
                    if method.feedback=='fert':
                        priorities*=.88
                        for i,r in enumerate(lineage):
                            for spec in selection or []:
                                if spec['demo_id']==r['source_demo_id']:
                                    priorities[i]+=max(0,min(r['source_frame']+50,spec['end'])-max(r['source_frame'],spec['start']))/50
                        priorities=np.maximum(priorities,1e-6);selection_log['priority_rule']='rho=.88 plus successful-A chunk overlap/50; no failed B actions'
                    dump(folder/f'feedback_{k:02d}.json',selection_log)
                    costs.append(dict(phase='selection_retarget',wall_seconds=time.monotonic()-update_start,added_anchors=0 if new_audit is None or new_data is None else new_audit['anchors']))
                    ckpt,learning,hit=train(run,data,ckpt,config['updates_per_rollout'],seed,k,new_start,priorities if method.feedback=='fert' else None)
                    updates+=config['updates_per_rollout'];costs.append(dict(phase='training',cache_hit=hit,**learning));costs[-1]['wall_seconds']=0. if hit else learning['wall_seconds']
                    progress('adaptation_checkpoint',dict(method=name,seed=seed,actual_rollouts=actual,updates=updates,anchors=len(lineage)))
                    if k in config['budgets']:records.append(checkpoint_record(run,folder,name,seed,k,actual,ckpt,data,updates,lineage,costs))
                for k in config['budgets']:
                    if not (folder/f'K_{k:02d}.json').exists():records.append(checkpoint_record(run,folder,name,seed,k,actual,ckpt,data,updates,lineage,costs,reason=reason or 'Adaptation terminated early'))
                dump(folder/'done.json',dict(status='completed' if reason is None else 'partial',method=name,wall_seconds=time.monotonic()-start,reason=reason))
        dump(run/'checkpoints.json',records);dump(run/'method_status.json',status)
        dump(run/'TRAINING_COMPLETE.json',dict(checkpoints=len(records),adaptation_rollouts=ledger.count('adaptation'),development_rollouts=ledger.count('development'),evaluation_rollouts=ledger.count('evaluation'),evaluation_used_for_selection=False))
    finally:pool.close()
