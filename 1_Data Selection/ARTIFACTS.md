# Stage 1 evidence, configuration, and tests

## Frozen fast-run evidence

The following files are from actual run `fast_20261105T182127Z`; links refer to published lightweight copies. Raw arrays/checkpoints remain in the local run tree.

- [01_selection/selection.json](../results/fast_20261105T182127Z/01_selection/selection.json)
- [01_selection/source_inspection.json](../results/fast_20261105T182127Z/01_selection/source_inspection.json)

[Run report](../results/fast_20261105T182127Z/FINAL_REPORT.md) · [run manifest](../results/fast_20261105T182127Z/manifest.json) · [source environment](../results/fast_20261105T182127Z/environment.json).

## Configuration

- [Bootstrap](../configs/bootstrap.yaml): minimal provided-seed pipeline.
- [Fast](../configs/fast.yaml): five selected source demos, ten target teachers if obtainable, bounded BC/evaluation.
- [Full](../configs/full.yaml): real human input required for CLIP/Qwen processing.

No human input means `human_clip_selection=skipped_missing_input`; it does not count as a successful CLIP experiment. Start-position diversity is a heuristic and does not establish visual or semantic demonstration quality.

## Verification

Relevant tests: [test_dataset_parser.py](../tests/test_dataset_parser.py), [test_stage_contracts.py](../tests/test_stage_contracts.py).

```bash
.venv/bin/python -m pytest tests/test_dataset_parser.py tests/test_stage_contracts.py -q
```

The [stage-entry tests](../tests/test_stage_entrypoints.py) validate this folder's wrapper routing without running simulation. Simulator tests are marked `sim`; integration/process tests are marked `integration`.

## Reproduction and provenance

Use the [detached runtime scripts](../scripts/) for long runs. Preserve the selected source IDs, task source, teacher origin, seed, model/controller, simulator versions, and `oracle_sim_state` flag when comparing episodes. A source-scripted Panda video, an actual source-dataset state replay, a source-retargeted UR5e teacher, and a learned-policy rollout are different evidence types.
