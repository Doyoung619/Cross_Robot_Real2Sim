"""Measured-only CSV, Wilson intervals, cost tables and standalone figures."""
from pathlib import Path
import csv
import json
import shutil

import numpy as np

from .core import METHODS,Ledger,dump,wilson,digest


def export(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);config=json.loads((run/'config.json').read_text());root=Path(__file__).resolve().parents[3]
    dest=root/'results'/run.name;dest.mkdir(parents=True,exist_ok=True);rows=[]
    aliases=json.loads((run/'evaluated_checkpoints.json').read_text()) if (run/'evaluated_checkpoints.json').exists() else []
    ledger=Ledger(run/'rollouts.jsonl');physical=ledger.events()
    for c in aliases:
        directory=run/'evaluations'/c['evaluation_key'];episodes=[json.loads(f.read_text()) for f in sorted(directory.glob('episode_*/result.json'))]
        n=len(episodes)
        if not n:continue
        success=sum(r['success'] for r in episodes);lo,hi=wilson(success,n)
        costs=c['cost_events'];training=[v for v in costs if 'cuda_update_elapsed_seconds' in v]
        owned=[r for r in episodes if r['method']==c['method'] and r['training_seed']==c['training_seed']]
        row=dict(method=c['method'],label=METHODS[c['method']].label,fidelity=METHODS[c['method']].fidelity,
            training_seed=c['training_seed'],budget_K=c['budget_K'],actual_adaptation_rollouts=c['actual_adaptation_rollouts'],development_rollouts=0,
            evaluated_states=n,requested_evaluated_states=c['requested_evaluation_trials'],evaluation_status=c['evaluation_status'],successes=success,
            success_rate=success/n,ci95_low=lo,ci95_high=hi,mean_goal_error_m=float(np.mean([r['goal_error_m'] for r in episodes])),
            mean_contact_fraction=float(np.mean([r['contact_fraction'] for r in episodes if r.get('contact_fraction') is not None])) if any(r.get('contact_fraction') is not None for r in episodes) else None,
            mean_eef_tracking_error_m=float(np.mean([r['mean_eef_tracking_error_m'] for r in episodes if r['mean_eef_tracking_error_m'] is not None])) if any(r['mean_eef_tracking_error_m'] is not None for r in episodes) else None,
            joint_limit_violation_steps=sum(r['joint_limit_violation_steps'] for r in episodes),arm_collision_steps=sum(r['arm_collision_steps'] for r in episodes),
            optimizer_updates=c['optimizer_updates'],source_demonstrations=len(c['unique_source_demos']),training_anchors=c['training_anchors'],
            adaptation_and_training_wall_s=sum(v.get('wall_seconds',0) for v in costs),
            cuda_update_elapsed_s=sum(v.get('cuda_update_elapsed_seconds',0) for v in training if not v.get('cache_hit')),
            equivalent_uncached_cuda_update_s=sum(v.get('cuda_update_elapsed_seconds',0) for v in training),
            adaptation_inference_cuda_s=sum(v.get('inference_cuda_elapsed_seconds',0) for v in costs),
            evaluation_inference_cuda_s=sum(r.get('inference_cuda_elapsed_seconds',0) for r in episodes),
            selection_retarget_wall_s=sum(v.get('wall_seconds',0) for v in costs if v['phase']=='selection_retarget'),
            peak_torch_allocated_bytes=max((v.get('peak_torch_allocated_bytes',0) for v in training),default=0),
            evaluation_wall_s=sum(r['wall_seconds'] for r in episodes),
            physical_evaluation_episodes_in_cache=sum(r['physical_rollout_count'] for r in episodes),
            evaluation_cache_key=c['evaluation_key'],checkpoint_sha256=c['checkpoint_sha256'],budget_shortfall_reason=c.get('budget_shortfall_reason'))
        rows.append(row)
    for row in rows:
        zero=next((r for r in rows if r['method']==row['method'] and r['training_seed']==row['training_seed'] and r['budget_K']==0),None)
        row['success_rate_gain_from_K0']=row['success_rate']-zero['success_rate'] if zero else None
    dump(dest/'metrics.json',rows)
    with (dest/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['method','budget_K','actual_adaptation_rollouts','evaluated_states','success_rate']);w.writeheader();w.writerows(rows)
    for name in ['config.json','protocol.json','method_status.json','TRAINING_COMPLETE.json','EVALUATION_FINISHED.json','status.json']:
        if (run/name).exists():shutil.copyfile(run/name,dest/name)
    for name in ['manifest.json','adaptation.json','evaluation.json','development.json']:
        if (run/'states'/name).exists():
            (dest/'states').mkdir(exist_ok=True);shutil.copyfile(run/'states'/name,dest/'states'/name)
    shutil.copyfile(run/'rollouts.jsonl',dest/'rollouts.jsonl') if (run/'rollouts.jsonl').exists() else None
    if (run/'shared/source_pool.json').exists():shutil.copyfile(run/'shared/source_pool.json',dest/'source_pool.json')
    # Small feedback provenance is exported; image arrays and weights stay local.
    for f in (run/'methods').rglob('feedback_*.json'):
        target=dest/'provenance'/f.relative_to(run/'methods');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    representatives={}
    for c in aliases:representatives[c['method'],c['training_seed']]=c
    for c in representatives.values():
        for f in sorted((run/'evaluations'/c['evaluation_key']).glob('episode_*/rollout.mp4'))[:1]:
            target=dest/'videos'/f"{c['method']}_seed{c['training_seed']}_K{c['budget_K']}_{f.parent.name}.mp4"
            target.parent.mkdir(exist_ok=True);shutil.copyfile(f,target)
    counts=dict(adaptation=ledger.count('adaptation'),development=ledger.count('development'),evaluation=ledger.count('evaluation'),smoke=ledger.count('smoke'),
        started_but_no_finished=sum(e['event']=='started' and not any(t['event']=='finished' and t['episode_id']==e['episode_id'] for t in physical) for e in physical))
    dump(dest/'interaction_counts.json',counts)
    if rows:
        fig,ax=plt.subplots(figsize=(11,6))
        for name in config['methods']:
            for seed in config['training_seeds']:
                values=[r for r in rows if r['method']==name and r['training_seed']==seed]
                if not values:continue
                # The x coordinate is ACTUAL interaction count, not the allowed budget.
                points={r['actual_adaptation_rollouts']:r for r in values};values=[points[k] for k in sorted(points)]
                x=[r['actual_adaptation_rollouts'] for r in values];y=[r['success_rate'] for r in values]
                label=METHODS[name].label+(f' (seed{seed})' if len(config['training_seeds'])>1 else '')
                if len(values)==1 and max(config['budgets'])>0:
                    ax.axhline(y[0],linestyle=':',alpha=.6,label=label+' [0-interaction reference]')
                    ax.scatter(x,y,s=25)
                else:ax.errorbar(x,y,yerr=[[r['success_rate']-r['ci95_low'] for r in values],[r['ci95_high']-r['success_rate'] for r in values]],marker='o',capsize=3,label=label)
        ax.set(xlabel='Actual B adaptation rollouts',ylabel='Success rate on sealed evaluation starts',ylim=(-.03,1.03),title=config['name']+' | 95% Wilson intervals per training seed')
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig(dest/'success_vs_rollouts.png',dpi=160,bbox_inches='tight');fig.savefig(dest/'success_vs_rollouts.pdf',bbox_inches='tight');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for name in config['methods']:
            v=sorted([r for r in rows if r['method']==name and r['training_seed']==config['training_seeds'][0]],key=lambda r:r['budget_K'])
            if not v:continue
            for ax,key in zip(axes,['mean_goal_error_m','equivalent_uncached_cuda_update_s','training_anchors']):ax.plot([r['actual_adaptation_rollouts'] for r in v],[r[key] for r in v],marker='o',label=METHODS[name].label)
        for ax,title in zip(axes,['Mean final XY goal error (m)','Equivalent CUDA update timeline (s)','Training anchors']):ax.set(xlabel='Actual B adaptation rollouts',title=title);ax.grid(alpha=.2)
        axes[-1].legend(fontsize=6,loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig(dest/'error_cost_data.png',dpi=160,bbox_inches='tight');plt.close(fig)
    text=['# Fixed-GR1 cross-embodiment adaptation benchmark: measured results','',
        f"Profile: **{config['name']}**. These are measured runs, not assumed performance. Training seeds: {config['training_seeds']}. Requested budgets: {config['budgets']}. Evaluation starts requested per checkpoint: {config['evaluation_states']}.",'',
        'All adaptation checkpoints are frozen before final evaluation. No final-evaluation observation, success, error or seed-specific result is supplied to source selection, retargeting, training, checkpoint selection or tuning. Development rollout budget is0; checkpoints are the last weights after a fixed training schedule. Primary success requires the native On predicate for20 consecutive20Hz control steps, matching the earlier protocol; final-step native predicate is also logged.','',
        'State banks use distinct native LIBERO initial placements, not repetitions of the two old scenes. The plate placement range is narrow (2cm by2cm), so this is in-distribution evaluation, not broad OOD or physical-robot generalization.','',
        f"Actual physical interaction counts: {counts}. Cached results of identical checkpoints are references to the same trials, not additional episodes. Static IK/rendering contributes0 B rollouts.",'',
        '## Measured checkpoints','', '| Method | Seed | Allowed K | Actual K | Evaluation n | Success | 95% CI | Mean XY error | Updates | Anchors |',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in rows:text.append(f"| {r['label']} | {r['training_seed']} | {r['budget_K']} | {r['actual_adaptation_rollouts']} | {r['evaluated_states']} | {r['successes']}/{r['evaluated_states']} | [{r['ci95_low']:.3f},{r['ci95_high']:.3f}] | {r['mean_goal_error_m']*100:.2f}cm | {r['optimizer_updates']} | {r['training_anchors']} |")
    text+=['','## What is not established','',
        'The hypothesis that failure-guided adaptation beats existing methods is not assumed. A smoke profile has short horizons/tiny evaluation samples and is only an execution check. An incomplete method/budget/seed matrix cannot establish the final research claim. Wilson intervals describe binomial uncertainty within one training seed, not variation across independently trained policies.','',
        'CEI-inspired, STRAP-style and FERT-inspired are common-interface adaptations, not official paper reproductions. See the benchmark README for preserved mechanisms and replacements. Source-conditioned synthetic actions do not verify target pushing dynamics. No failed B actions become labels; no hidden B teacher collection is performed.','',
        'CUDA-event elapsed timing is a device timeline proxy, not exact GPU-active compute. Peak Torch allocation excludes some EGL/driver allocations. Shared source inspection/embedding preparation, actual cache savings and uncached-equivalent training costs must be read separately.','',
        '[CSV](metrics.csv) · [JSON](metrics.json) · [Interaction ledger](rollouts.jsonl) · [Counts](interaction_counts.json)','']
    if (dest/'success_vs_rollouts.png').exists():text+=['![Measured learning curves](success_vs_rollouts.png)','']
    text+=['## Exported videos','']+[f'- [{f.name}](videos/{f.name})' for f in sorted((dest/'videos').glob('*.mp4'))]
    (dest/'README.md').write_text('\n'.join(text)+'\n')
    files=[dict(path=str(f.relative_to(dest)),bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(dest.rglob('*')) if f.is_file() and f.name!='artifact_manifest.json']
    dump(dest/'artifact_manifest.json',dict(files=files));return dest
