# 1. Data Selection

Choose coherent source Panda demonstrations and, when supplied, relevant real human clips for the can-inside-bin task.

The executed fast run selected five real Panda demonstrations (`demo_3`, `demo_5`, `demo_7`, `demo_4`, `demo_1`) from eight inspected candidates. No human clip was supplied; CLIP selection is implemented but untested on these recorded runs.

## Run this stage

From the repository root, after setup:

```bash
.venv/bin/python "1_Data Selection/run.py" --help
.venv/bin/python "1_Data Selection/run.py" --config configs/fast.yaml --run-id RUN_ID
```

Stage 1 can start a new run ID. Use the same ID and unchanged configuration for later stages. Add `--resume` to retain this stage only when its saved contracts and hashes still validate. These wrappers run one stage in the foreground; for long work over SSH, use `bash scripts/launch.sh fast` to run the integrated pipeline in detached tmux.

## Code map

The executable algorithms remain in the canonical `src/cross_robot_real2sim/stage1_selection/` package. This top-level folder is the stage's documentation and executable navigation entry, with no duplicated algorithm implementation or symlinks.

| Canonical file | Responsibility |
|---|---|
| [run.py](../src/cross_robot_real2sim/stage1_selection/run.py) | Stage dispatch and selection provenance |
| [metadata_baseline.py](../src/cross_robot_real2sim/stage1_selection/metadata_baseline.py) | Shape/robot checks and deterministic diverse-start selection |
| [clip_ranker.py](../src/cross_robot_real2sim/stage1_selection/clip_ranker.py) | Optional real-frame CLIP scoring |

Shared contracts: [schemas](../src/cross_robot_real2sim/schemas.py), [robot/source adapters](../src/cross_robot_real2sim/adapters/), and [orchestrator](../src/cross_robot_real2sim/orchestrator/).

## Inputs and outputs

Input: Task text, robot/task metadata, the minimal robomimic Can/PH HDF5, and optional real clips in `data/human/`.

Output: `01_selection/selection.json` and `01_selection/source_inspection.json`. Raw executions live in `runs/RUN_ID/`; lightweight frozen evidence is published under `results/`.

Read [methods and limitations](METHODS.md) and the [artifact/configuration/test index](ARTIFACTS.md). Return to the [five-stage overview](../README.md).
