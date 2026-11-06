# 5. Failure Analysis

Turn real rollout evidence into conservative diagnostic rules, feedback requests, and one bounded corrective collection/retraining comparison.

Fast BC remained 9/10 before and after three extra teachers, so no success-rate improvement is claimed. Bootstrap improved from 0/3 to 1/3 after two additional successful teachers. Both comparisons use identical before/after held-out seeds.

## Run this stage

From the repository root, after setup:

```bash
.venv/bin/python "5_Failure Analysis/run.py" --help
.venv/bin/python "5_Failure Analysis/run.py" --config configs/fast.yaml --run-id RUN_ID
```

Stages 1–4 must already have written the required artifacts under the same run ID and configuration. Add `--resume` to retain this stage only when its saved contracts and hashes still validate. These wrappers run one stage in the foreground; for long work over SSH, use `bash scripts/launch.sh fast` to run the integrated pipeline in detached tmux.

## Code map

The executable algorithms remain in the canonical `src/cross_robot_real2sim/stage5_diagnosis/` package. This top-level folder is the stage's documentation and executable navigation entry, with no duplicated algorithm implementation or symlinks.

| Canonical file | Responsibility |
|---|---|
| [run.py](../src/cross_robot_real2sim/stage5_diagnosis/run.py) | Diagnosis, fresh feedback state, and comparison assembly |
| [detectors.py](../src/cross_robot_real2sim/stage5_diagnosis/detectors.py) | Evidence-based simulator diagnostic rules |
| [feedback.py](../src/cross_robot_real2sim/stage5_diagnosis/feedback.py) | Ranked feedback requests and supported bottleneck hypotheses |
| [retrain_check.py](../src/cross_robot_real2sim/stage5_diagnosis/retrain_check.py) | Bounded extra collection/retraining with the same evaluation seeds |

Shared contracts: [schemas](../src/cross_robot_real2sim/schemas.py), [robot/source adapters](../src/cross_robot_real2sim/adapters/), and [orchestrator](../src/cross_robot_real2sim/orchestrator/).

## Inputs and outputs

Input: Actual Stage 3 teacher attempts, Stage 4 learned rollouts/evaluations, simulator goal predicates, and explicit intervention budgets.

Output: `05_diagnosis/failure_report.csv`, `feedback_requests.json`, `before_after.json`, fresh feedback-run artifacts, and labeled comparisons. Raw executions live in `runs/RUN_ID/`; lightweight frozen evidence is published under `results/`.

Read [methods and limitations](METHODS.md) and the [artifact/configuration/test index](ARTIFACTS.md). Return to the [five-stage overview](../README.md).
