# 3. Cross-Robot Retargeting

Extract limited source-relative grasp/lift geometry and execute target UR5e demonstrations through its own closed-loop simulator controller.

The recorded fast run generated 10/10 successful source-retargeted UR5e teachers and executed a separate live scripted Panda branch. The teachers derive from five selected Panda demos, not copied source joint commands or a pretrained source policy.

## Run this stage

From the repository root, after setup:

```bash
.venv/bin/python "3_Cross-Robot-Retargeting/run.py" --help
.venv/bin/python "3_Cross-Robot-Retargeting/run.py" --config configs/fast.yaml --run-id RUN_ID
```

Stages 1–2 must already have written the required artifacts under the same run ID and configuration. Add `--resume` to retain this stage only when its saved contracts and hashes still validate. These wrappers run one stage in the foreground; for long work over SSH, use `bash scripts/launch.sh fast` to run the integrated pipeline in detached tmux.

## Code map

The executable algorithms remain in the canonical `src/cross_robot_real2sim/stage3_retarget/` package. This top-level folder is the stage's documentation and executable navigation entry, with no duplicated algorithm implementation or symlinks.

| Canonical file | Responsibility |
|---|---|
| [run.py](../src/cross_robot_real2sim/stage3_retarget/run.py) | Bounded source/target collection and preserved attempt evidence |
| [source_trace.py](../src/cross_robot_real2sim/stage3_retarget/source_trace.py) | Actual EEF/object trace extraction and source grasp/lift summary |
| [object_frame.py](../src/cross_robot_real2sim/stage3_retarget/object_frame.py) | Explicit frame transformations |
| [waypoints.py](../src/cross_robot_real2sim/stage3_retarget/waypoints.py) | Source-derived offsets and target grip-site calibration |
| [controller.py](../src/cross_robot_real2sim/stage3_retarget/controller.py) | Observable phase tracker and shared features |
| [rollouts.py](../src/cross_robot_real2sim/stage3_retarget/rollouts.py) | Executed simulator rollout collection and provenance |

Shared contracts: [schemas](../src/cross_robot_real2sim/schemas.py), [robot/source adapters](../src/cross_robot_real2sim/adapters/), and [orchestrator](../src/cross_robot_real2sim/orchestrator/).

## Inputs and outputs

Input: Validated Stage 2 task, Stage 1 selected HDF5 demos, and live Panda/UR5e simulator state.

Output: `03_retarget/source_trace.npz`, per-demo trace metadata, `attempts.csv`, executed rollout NPZs, and labeled evidence videos. Raw executions live in `runs/RUN_ID/`; lightweight frozen evidence is published under `results/`.

Read [methods and limitations](METHODS.md) and the [artifact/configuration/test index](ARTIFACTS.md). Return to the [five-stage overview](../README.md).
