# 4. Policy Learning

Duration follow-up: [600 seconds of additional B LoRA training](../results/vla_extra10m_20261113T193618Z/README.md), starting from the initial 100-update adapter, added 8,130 updates on the same source-derived data. No B execution demonstrations were added. Success stayed 0/2 and goal error worsened from 10.95 to 20.00 cm. Parent optimizer state was unavailable, so adapter weights continued with fresh AdamW; the new run saves optimizer/RNG state.

Latest [source-only VLA pilot](../results/vla_minimal_20261113T185314Z/README.md): strictly load the existing 450M SmolVLA, freeze its base weights, and update 334,592 rank-eight LoRA parameters for 100 steps on source-derived synthetic observations/actions. No executed B demonstrations enter training. See [VLA learning code](../scripts/vla_pilot_learning.py) and [saved adapter](../results/vla_minimal_20261113T185314Z/adapter.safetensors). The wrapper below runs the historical state-MLP baseline, not this VLA experiment.

Learn target task-space actions only from successfully executed UR5e teachers, then evaluate the saved model in fresh closed-loop episodes.

The recorded fast checkpoint was trained on 10 successful UR5e teachers derived from five Panda demos and achieved 9/10 held-out placements. The English case studies show this shared frozen checkpoint across different episode seeds; each displayed source clip is context, not a separate per-case training checkpoint.

## Run this stage

From the repository root, after setup:

```bash
.venv/bin/python "4_Policy Learning/run.py" --help
.venv/bin/python "4_Policy Learning/run.py" --config configs/fast.yaml --run-id RUN_ID
```

Stages 1–3 must already have written the required artifacts under the same run ID and configuration. Add `--resume` to retain this stage only when its saved contracts and hashes still validate. These wrappers run one stage in the foreground; for long work over SSH, use `bash scripts/launch.sh fast` to run the integrated pipeline in detached tmux.

## Code map

The executable algorithms remain in the canonical `src/cross_robot_real2sim/stage4_policy/` package. This top-level folder is the stage's documentation and executable navigation entry, with no duplicated algorithm implementation or symlinks.

| Canonical file | Responsibility |
|---|---|
| [run.py](../src/cross_robot_real2sim/stage4_policy/run.py) | Dataset → training → fresh-process evaluation |
| [build_dataset.py](../src/cross_robot_real2sim/stage4_policy/build_dataset.py) | Successful-teacher filtering, feature checks, and disjoint seeds |
| [bc_model.py](../src/cross_robot_real2sim/stage4_policy/bc_model.py) | Two-hidden-layer MLP and saved-policy inference |
| [train.py](../src/cross_robot_real2sim/stage4_policy/train.py) | XYZ MSE + gripper BCE optimization and checkpoint state |
| [evaluate.py](../src/cross_robot_real2sim/stage4_policy/evaluate.py) | Actual fresh-process UR5e closed-loop evaluation |

Shared contracts: [schemas](../src/cross_robot_real2sim/schemas.py), [robot/source adapters](../src/cross_robot_real2sim/adapters/), and [orchestrator](../src/cross_robot_real2sim/orchestrator/).

## Inputs and outputs

Input: Successful Stage 3 UR5e rollout features/actions, their origin/source IDs, and a disjoint held-out seed set.

Output: `04_policy/train_data.npz`, normalization/provenance, `checkpoint.pt`, training log, fresh-process evaluation CSV/metrics, and learned-policy videos. Raw executions live in `runs/RUN_ID/`; lightweight frozen evidence is published under `results/`.

Read [methods and limitations](METHODS.md) and the [artifact/configuration/test index](ARTIFACTS.md). Return to the [five-stage overview](../README.md).
