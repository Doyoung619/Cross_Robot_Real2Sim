#!/usr/bin/env python3
"""Read-only analysis of an exported benchmark; never supplies adaptation feedback."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run, output):
    rows = json.loads((output / 'metrics.json').read_text())
    aliases_path = run / 'evaluated_checkpoints.json'
    aliases = json.loads(aliases_path.read_text()) if aliases_path.exists() else []
    summaries = []
    for checkpoint in aliases:
        directory = run / 'evaluations' / checkpoint['evaluation_key']
        episodes = [json.loads(p.read_text()) for p in sorted(directory.glob('episode_*/result.json'))]
        counts = Counter()
        for episode in episodes:
            if episode['success']:
                counts['success'] += 1
                continue
            counts['task_failed'] += 1
            if episode.get('contact_fraction') is not None and episode['contact_fraction'] < .01:
                counts['very_low_contact'] += 1
            if episode.get('mean_eef_tracking_error_m') is not None and episode['mean_eef_tracking_error_m'] > .03:
                counts['high_tracking_error'] += 1
            if episode.get('joint_limit_violation_steps', 0):
                counts['joint_limit_violation'] += 1
            if episode.get('arm_collision_steps', 0):
                counts['arm_environment_collision'] += 1
        summaries.append(dict(method=checkpoint['method'], training_seed=checkpoint['training_seed'],
                              budget_K=checkpoint['budget_K'], evaluated_states=len(episodes),
                              **{key: counts[key] for key in ['success', 'task_failed', 'very_low_contact',
                                 'high_tracking_error', 'joint_limit_violation', 'arm_environment_collision']}))
    write_csv(output / 'failure_indicators.csv', summaries)
    ablations = []
    for row in rows:
        if row['method'] not in {'fixed_source_train', 'selection_only', 'retarget_only', 'our_full'}:
            continue
        reference = next((candidate for candidate in rows
                          if candidate['method'] == 'fixed_source_train'
                          and candidate['training_seed'] == row['training_seed']
                          and candidate['budget_K'] == row['budget_K']), None)
        ablations.append(dict(method=row['method'], training_seed=row['training_seed'],
                              budget_K=row['budget_K'], actual_K=row['actual_adaptation_rollouts'],
                              n=row['evaluated_states'], success_rate=row['success_rate'],
                              goal_error_m=row['mean_goal_error_m'], anchors=row['training_anchors'],
                              delta_success_vs_fixed_source=(row['success_rate'] - reference['success_rate'])
                              if reference else None))
    write_csv(output / 'stage_ablations.csv', ablations)
    (output / 'ANALYSIS.md').write_text(
        '# Descriptive evaluation analysis\n\n'
        'These files analyze frozen final evaluations after adaptation. They are not inputs to training, '
        'retrieval, retargeting or checkpoint selection. The failure indicators overlap and are heuristic '
        'symptoms, not causal diagnoses: contact fraction <0.01; mean EEF tracking error >0.03 m; '
        'any logged joint-limit violation or arm/environment collision. Missing measurements are not '
        'classified as positive symptoms.\n\n'
        '[Failure indicators](failure_indicators.csv) · [Stage ablations](stage_ablations.csv)\n\n'
        'Ablation deltas compare the matched fixed-source control at the same allowed K and training seed; '
        'they are descriptive differences without a significance claim. The smoke horizon and two starts '
        'cannot establish adaptation quality. Check metrics.csv for actual counts, intervals and costs.\n\n'
        'Cost bars show cumulative adaptation/training wall time and measured CUDA update timeline. '
        'The latter is not exclusive GPU-active time. Identical training cache hits cost zero actual new '
        'CUDA updates; equivalent uncached compute is separately retained in metrics.csv. Shared preparation '
        'cost remains in protocol.json.\n')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        latest = {}
        for row in rows:
            key = (row['method'], row['training_seed'])
            if key not in latest or latest[key]['budget_K'] < row['budget_K']:
                latest[key] = row
        values = list(latest.values())
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(values) * .35)))
        labels = [f"{r['label']} / seed{r['training_seed']} / K={r['budget_K']}" for r in values]
        for ax, key, title in zip(axes, ['adaptation_and_training_wall_s', 'cuda_update_elapsed_s'],
                                  ['Adaptation/training wall time (s)', 'Actual CUDA update timeline (s)']):
            ax.barh(range(len(values)), [row[key] for row in values])
            ax.set_yticks(range(len(values)), labels if ax is axes[0] else [])
            ax.set_title(title)
            ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        fig.savefig(output / 'wall_cuda_cost.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_dir, args.output_dir)
