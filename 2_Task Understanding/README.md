# 2. Task Understanding

Produce a validated can-inside-bin task contract, preserve unknown physical quantities, and declare how semantic labels map into the simulator.

Recorded bootstrap/fast experiments validated a provided task seed. They did not infer a task from human video. Full mode explicitly requires a real clip and retains raw Qwen output for inspection.

## Run this stage

From the repository root, after setup:

```bash
.venv/bin/python "2_Task Understanding/run.py" --help
.venv/bin/python "2_Task Understanding/run.py" --config configs/fast.yaml --run-id RUN_ID
```

Stages 1–1 must already have written the required artifacts under the same run ID and configuration. Add `--resume` to retain this stage only when its saved contracts and hashes still validate. These wrappers run one stage in the foreground; for long work over SSH, use `bash scripts/launch.sh fast` to run the integrated pipeline in detached tmux.

## Code map

The executable algorithms remain in the canonical `src/cross_robot_real2sim/stage2_task/` package. This top-level folder is the stage's documentation and executable navigation entry, with no duplicated algorithm implementation or symlinks.

| Canonical file | Responsibility |
|---|---|
| [run.py](../src/cross_robot_real2sim/stage2_task/run.py) | Mode handling and task provenance |
| [seed_baseline.py](../src/cross_robot_real2sim/stage2_task/seed_baseline.py) | Provided-seed loading and validation |
| [qwen_vl.py](../src/cross_robot_real2sim/stage2_task/qwen_vl.py) | Optional Qwen video-frame interpretation and JSON parsing |
| [sam2_optional.py](../src/cross_robot_real2sim/stage2_task/sam2_optional.py) | Explicit unimplemented optional segmentation boundary |

Shared contracts: [schemas](../src/cross_robot_real2sim/schemas.py), [robot/source adapters](../src/cross_robot_real2sim/adapters/), and [orchestrator](../src/cross_robot_real2sim/orchestrator/).

## Inputs and outputs

Input: Stage 1 selection plus task text and either the provided task seed or actual selected human frames.

Output: `02_task/task_spec.json`, and sampled/annotated frames plus raw model text when perception executes. Raw executions live in `runs/RUN_ID/`; lightweight frozen evidence is published under `results/`.

Read [methods and limitations](METHODS.md) and the [artifact/configuration/test index](ARTIFACTS.md). Return to the [five-stage overview](../README.md).
