#!/usr/bin/env python3
"""Bounded development-rollout feedback into source selection and retargeting."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

from vla_pilot_common import ROOT,bootstrap,write_json


def fit(out, checkpoint, optimizer_path, updates, cumulative, progress, seed):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from vla_pilot_learning import load_model,InputContract
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    started=time.monotonic();model=load_model(True,checkpoint);contract=InputContract(model)
    with np.load(out/'processed_source_data.npz',allow_pickle=False) as f:
        images=f['images'].copy();states=f['states'].copy();actions=f['actions'].copy()
        masks=f['actions_id_pad'].copy() if 'actions_id_pad' in f else np.zeros(actions.shape[:2],bool)
    assert len(images)==len(states)==len(actions) and states.shape[1]==8 and actions.shape[1:]==(50,7)
    assert np.isfinite(states).all() and np.isfinite(actions).all() and np.abs(actions).max()<=1.001
    metadata=json.loads((out/'processing_audit.json').read_text()) if (out/'processing_audit.json').exists() else {}
    new_start=int(metadata.get('new_rows_start',len(states)))
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert all('lora_' in n for n,p in parameters)
    optimizer=torch.optim.AdamW([p for n,p in parameters],lr=1e-4)
    restored=False
    if optimizer_path and Path(optimizer_path).exists():
        optimizer.load_state_dict(torch.load(optimizer_path,map_location='cuda',weights_only=False)['optimizer'])
        restored=True
    model.train();logs=[]
    for step in range(updates):
        if 0<new_start<len(states):
            ids=np.r_[rng.integers(0,new_start,2),rng.integers(new_start,len(states),2)]
        else:ids=rng.integers(0,len(states),4)
        batch=contract.batch(images[ids],states[ids],actions[ids])
        batch['actions_id_pad']=torch.as_tensor(masks[ids],device='cuda',dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=model(batch)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite feedback training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for n,p in parameters],1.);optimizer.step()
        if step%25==0 or step==updates-1:
            row=dict(step=step+1,total_step=cumulative+step+1,loss=float(loss.detach()))
            logs.append(row);progress('feedback_training',dict(round=out.name,**row))
    weights={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if 'lora_' in k}
    save_file(weights,str(out/'adapter.safetensors'))
    torch.save({'optimizer':optimizer.state_dict()},out/'optimizer.pt')
    result=dict(updates=updates,steps=cumulative+updates,cumulative_steps=cumulative+updates,
                training_frames=len(states),new_rows_start=new_start,new_rows_count=len(states)-new_start,
                sampling='half_previous_source_half_new_corrective_source_when_available',
                optimizer_state_restored=restored,parent_adapter=str(checkpoint),
                parent_adapter_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256((out/'adapter.safetensors').read_bytes()).hexdigest(),
                executed_B_training_demonstrations=0,B_rollout_actions_used_as_targets=False,
                padded_future_actions_masked=True,trainable_parameters=sum(p.numel() for n,p in parameters),
                batch_size=4,learning_rate=1e-4,rank=8,duration_seconds=time.monotonic()-started,final_loss=logs[-1]['loss'])
    write_json(out/'training.json',result);write_json(out/'train_log.json',logs)
    del optimizer,model,contract,parameters;torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--deadline',required=True)
    args=parser.parse_args();out=Path(args.run_dir).resolve();request=json.loads((out/'request.json').read_text())
    parent=ROOT/request['parent_run'];deadline=datetime.fromisoformat(args.deadline.replace('Z','+00:00')).timestamp()
    ledger=dict(started_at=datetime.now(timezone.utc).isoformat(),events=[])
    summary=dict(protocol='B_development_feedback_changes_A_selection_and_offline_retargeting',
                 development_source_ids=request['development_source_ids'],final_source_ids=request['final_source_ids'],
                 parent_run=str(parent),rounds=[],executed_B_training_demonstrations=0)
    def progress(stage,details):
        event=dict(stage=stage,at=datetime.now(timezone.utc).isoformat(),details=details)
        ledger['events'].append(event);ledger['current_stage']=stage;write_json(out/'progress.json',ledger)
        print(json.dumps(event,default=str),flush=True)
    def prepare(folder):
        folder.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/'camera_calibration.npz',folder/'camera_calibration.npz')
        selected=json.loads((parent/'selection.json').read_text());selected['evaluation_source_ids']=request['development_source_ids']
        write_json(folder/'selection.json',selected)
    def measured(folder,checkpoint,label,ids):
        if (folder/'evaluation.json').exists():return json.loads((folder/'evaluation.json').read_text())
        return evaluate_feedback(folder,checkpoint,label,ids,progress,deadline,horizon=220)
    def score(rows):return (sum(r['success'] for r in rows),-sum(r['goal_error_m'] for r in rows)/len(rows))
    try:
        bootstrap()
        from vla_feedback_rollouts import evaluate_feedback,diagnose
        baseline=out/'rounds/baseline';prepare(baseline)
        checkpoint=parent/'adapter.safetensors'
        rows=measured(baseline,checkpoint,'baseline_100',request['development_source_ids'])
        diagnostic=diagnose(rows);write_json(baseline/'diagnosis.json',diagnostic)
        write_json(baseline/'intervention.json',dict(description='Original 100-update adapter, before any feedback.'))
        best=dict(folder=str(baseline),checkpoint=str(checkpoint),rows=rows,score=score(rows),round=0)
        previous=parent;optimizer_path=None
        for round_index in range(1,request['rounds']+1):
            if time.time()>deadline-240:break
            module=ROOT/'scripts/vla_feedback_processing.py'
            while not module.exists() and time.time()<deadline-240:time.sleep(1)
            from vla_feedback_processing import process_feedback
            current=out/f'rounds/round_{round_index}';prepare(current)
            write_json(current/'input_diagnosis.json',diagnostic)
            progress('feedback_requested',dict(round=round_index,diagnostic=diagnostic))
            if not (current/'processed_source_data.npz').exists():
                metadata=process_feedback(current,previous,diagnostic,round_index,progress)
            else:metadata=json.loads((current/'processing_audit.json').read_text())
            write_json(current/'intervention.json',dict(description=f"Measured {diagnostic['dominant_hypothesis']} feeds source retrieval, bounded static contact optimization and full action-chunk recalculation.",diagnosis=diagnostic,processing=metadata))
            if not (current/'training.json').exists():
                training=fit(current,checkpoint,optimizer_path,100,100*round_index,progress,seed=80+round_index)
            else:training=json.loads((current/'training.json').read_text())
            checkpoint=current/'adapter.safetensors';optimizer_path=current/'optimizer.pt'
            rows=measured(current,checkpoint,f'feedback_round_{round_index}',request['development_source_ids'])
            diagnostic=diagnose(rows);write_json(current/'diagnosis.json',diagnostic)
            record=dict(round=round_index,folder=str(current),training=training,
                        success_count=sum(r['success'] for r in rows),mean_goal_error_m=sum(r['goal_error_m'] for r in rows)/len(rows),
                        diagnosis=diagnostic,improved_over_best=score(rows)>tuple(best['score']))
            summary['rounds'].append(record)
            if score(rows)>tuple(best['score']):best=dict(folder=str(current),checkpoint=str(checkpoint),rows=rows,score=score(rows),round=round_index)
            summary['best_development_round']=best['round'];write_json(out/'summary.json',summary)
            previous=current
        if time.time()<deadline-180:
            control=out/'control';prepare(control)
            if not (control/'processed_source_data.npz').exists():(control/'processed_source_data.npz').symlink_to(parent/'processed_source_data.npz')
            write_json(control/'intervention.json',dict(description='Compute-matched control: same extra update budget on unchanged original 189 anchors, no feedback selection or retargeting.'))
            if not (control/'training.json').exists():fit(control,parent/'adapter.safetensors',None,100*len(summary['rounds']),100,progress,seed=81)
            measured(control,control/'adapter.safetensors','no_feedback_control',request['development_source_ids'])
        final=out/'final';prepare(final);final_rows=[]
        for label,ckpt in [('final_original100',parent/'adapter.safetensors'),(f"final_selected_round_{best['round']}",Path(best['checkpoint']))]:
            final_rows.extend(evaluate_feedback(final,ckpt,label,request['final_source_ids'],progress,deadline,horizon=220))
        write_json(final/'evaluation.json',final_rows)
        summary.update(selected_round=best['round'],selected_checkpoint=best['checkpoint'],
                       final_comparison=final_rows,finished_at=datetime.now(timezone.utc).isoformat())
        write_json(out/'summary.json',summary)
        progress('packaging',dict(rounds=len(summary['rounds'])))
        from package_vla_feedback import package
        package(out)
        progress('completed',dict(summary=str(out/'summary.json')))
    except Exception:
        error=traceback.format_exc();progress('failed',dict(traceback=error));write_json(out/'summary.json',summary)
        (out/'FINAL_REPORT.md').write_text('# Feedback loop partial failure\n\n```text\n'+error+'\n```\n')
        raise


if __name__=='__main__':main()
