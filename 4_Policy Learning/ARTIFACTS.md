# Stage 4 evidence, configuration, and tests

## Frozen fast-run evidence

The following files are from actual run `fast_20261105T182127Z`; links refer to published lightweight copies. Raw arrays/checkpoints remain in the local run tree.

- [04_policy/dataset_metadata.json](../results/fast_20261105T182127Z/04_policy/dataset_metadata.json)
- [04_policy/train_log.csv](../results/fast_20261105T182127Z/04_policy/train_log.csv)
- [04_policy/train_metrics.json](../results/fast_20261105T182127Z/04_policy/train_metrics.json)
- [04_policy/eval.csv](../results/fast_20261105T182127Z/04_policy/eval.csv)
- [04_policy/eval_metrics.json](../results/fast_20261105T182127Z/04_policy/eval_metrics.json)

[Run report](../results/fast_20261105T182127Z/FINAL_REPORT.md) · [run manifest](../results/fast_20261105T182127Z/manifest.json) · [source environment](../results/fast_20261105T182127Z/environment.json).

## Configuration

- [Bootstrap](../configs/bootstrap.yaml): minimal provided-seed pipeline.
- [Fast](../configs/fast.yaml): five selected source demos, ten target teachers if obtainable, bounded BC/evaluation.
- [Full](../configs/full.yaml): real human input required for CLIP/Qwen processing.

Fast performance is 9/10 on ten simulator seeds, with privileged state and hand-coded online phase machinery. This is small-sample evidence and does not establish visual generalization or real-world transfer. Three additional fast teachers preserved the 9/10 success count and slightly worsened mean placement error.

## Verification

Relevant tests: [test_policy_learning.py](../tests/test_policy_learning.py), [test_stale_stage_outputs.py](../tests/test_stale_stage_outputs.py), [test_end_to_end_bootstrap.py](../tests/test_end_to_end_bootstrap.py).

```bash
.venv/bin/python -m pytest tests/test_policy_learning.py tests/test_stale_stage_outputs.py tests/test_end_to_end_bootstrap.py -q
```

The [stage-entry tests](../tests/test_stage_entrypoints.py) validate this folder's wrapper routing without running simulation. Simulator tests are marked `sim`; integration/process tests are marked `integration`.

## Reproduction and provenance

Use the [detached runtime scripts](../scripts/) for long runs. Preserve the selected source IDs, task source, teacher origin, seed, model/controller, simulator versions, and `oracle_sim_state` flag when comparing episodes. A source-scripted Panda video, an actual source-dataset state replay, a source-retargeted UR5e teacher, and a learned-policy rollout are different evidence types.
