# Stage 3 evidence, configuration, and tests

## Frozen fast-run evidence

The following files are from actual run `fast_20261105T182127Z`; links refer to published lightweight copies. Raw arrays/checkpoints remain in the local run tree.

- [03_retarget/attempts.csv](../results/fast_20261105T182127Z/03_retarget/attempts.csv)
- [03_retarget/stats.json](../results/fast_20261105T182127Z/03_retarget/stats.json)

[Run report](../results/fast_20261105T182127Z/FINAL_REPORT.md) · [run manifest](../results/fast_20261105T182127Z/manifest.json) · [source environment](../results/fast_20261105T182127Z/environment.json).

## Configuration

- [Bootstrap](../configs/bootstrap.yaml): minimal provided-seed pipeline.
- [Fast](../configs/fast.yaml): five selected source demos, ten target teachers if obtainable, bounded BC/evaluation.
- [Full](../configs/full.yaml): real human input required for CLIP/Qwen processing.

This is source-grounded waypoint retargeting with hand-coded phases, calibrated grip offsets, and privileged state. It is not full trajectory retargeting, learned source-policy transfer, or an ablation showing source geometry is necessary. The source orientation trace is stored, while target control holds the initial downward grip orientation.

## Verification

Relevant tests: [test_action_adapter.py](../tests/test_action_adapter.py), [test_goal_evaluation.py](../tests/test_goal_evaluation.py), [test_sim_smoke.py](../tests/test_sim_smoke.py).

```bash
.venv/bin/python -m pytest tests/test_action_adapter.py tests/test_goal_evaluation.py tests/test_sim_smoke.py -q
```

The [stage-entry tests](../tests/test_stage_entrypoints.py) validate this folder's wrapper routing without running simulation. Simulator tests are marked `sim`; integration/process tests are marked `integration`.

## Reproduction and provenance

Use the [detached runtime scripts](../scripts/) for long runs. Preserve the selected source IDs, task source, teacher origin, seed, model/controller, simulator versions, and `oracle_sim_state` flag when comparing episodes. A source-scripted Panda video, an actual source-dataset state replay, a source-retargeted UR5e teacher, and a learned-policy rollout are different evidence types.
