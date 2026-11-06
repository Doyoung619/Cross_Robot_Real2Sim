# Stage 5 evidence, configuration, and tests

## Frozen fast-run evidence

The following files are from actual run `fast_20261105T182127Z`; links refer to published lightweight copies. Raw arrays/checkpoints remain in the local run tree.

- [05_diagnosis/failure_report.csv](../results/fast_20261105T182127Z/05_diagnosis/failure_report.csv)
- [05_diagnosis/feedback_requests.json](../results/fast_20261105T182127Z/05_diagnosis/feedback_requests.json)
- [05_diagnosis/before_after.json](../results/fast_20261105T182127Z/05_diagnosis/before_after.json)

[Run report](../results/fast_20261105T182127Z/FINAL_REPORT.md) · [run manifest](../results/fast_20261105T182127Z/manifest.json) · [source environment](../results/fast_20261105T182127Z/environment.json).

## Configuration

- [Bootstrap](../configs/bootstrap.yaml): minimal provided-seed pipeline.
- [Fast](../configs/fast.yaml): five selected source demos, ten target teachers if obtainable, bounded BC/evaluation.
- [Full](../configs/full.yaml): real human input required for CLIP/Qwen processing.

The corrective iteration is a bounded heuristic experiment. Fast success stayed 9/10 and mean XY error rose from about 0.0815 m to 0.0846 m; there is no measured fast improvement. Small bootstrap gains do not establish statistical significance. No learned causal diagnosis or human feedback loop is implemented.

## Verification

Relevant tests: [test_goal_evaluation.py](../tests/test_goal_evaluation.py), [test_policy_learning.py](../tests/test_policy_learning.py), [test_stale_stage_outputs.py](../tests/test_stale_stage_outputs.py), [test_resume_integrity.py](../tests/test_resume_integrity.py).

```bash
.venv/bin/python -m pytest tests/test_goal_evaluation.py tests/test_policy_learning.py tests/test_stale_stage_outputs.py tests/test_resume_integrity.py -q
```

The [stage-entry tests](../tests/test_stage_entrypoints.py) validate this folder's wrapper routing without running simulation. Simulator tests are marked `sim`; integration/process tests are marked `integration`.

## Reproduction and provenance

Use the [detached runtime scripts](../scripts/) for long runs. Preserve the selected source IDs, task source, teacher origin, seed, model/controller, simulator versions, and `oracle_sim_state` flag when comparing episodes. A source-scripted Panda video, an actual source-dataset state replay, a source-retargeted UR5e teacher, and a learned-policy rollout are different evidence types.
