# Cross-Robot Real2Sim

A runnable five-stage research MVP for **can pick → lift → transport → place**, transferring source-grounded task-space geometry from **Franka Panda** demonstrations to **UR5e** in robosuite/MuJoCo `PickPlaceCan`.

The baseline executes real simulator rollouts, trains a small behavior-cloning policy on successful UR5e teacher episodes, evaluates the saved policy on held-out seeds, and performs one bounded feedback iteration. It uses **privileged simulator object poses, goals, and grasp-contact state** (`oracle_sim_state`). It does not establish vision-only or physical-robot transfer.

## Start and monitor

On headless Linux with NVIDIA GPU 0, Python 3.10, `tmux`, and `flock` available:

```bash
cd /home/dheo/Cross_Robot_Real2Sim
bash scripts/setup.sh
bash scripts/preflight.sh
bash scripts/launch.sh bootstrap
```

Setup creates the isolated `.venv`, installs pinned simulator/BC dependencies, and downloads **only** robomimic `v1.5/can/ph/low_dim_v15.hdf5` (46,889,752 bytes, 200 episodes). Bootstrap setup does not install or download CLIP/Qwen models. The tested stack uses robosuite 1.5.2, MuJoCo 3.3.7, PyTorch 2.6.0+cu124, Python 3.10.12, and EGL rendering. Exact installed versions are recorded in `requirements-working-bootstrap.lock` and each run's `environment.json`.

Launch prints the run ID, detached session, and log path. After it returns successfully, the job continues if SSH or VS Code disconnects. **The host must stay running; tmux does not provide reboot recovery.** The scripts activate the environment inside the detached session, use absolute paths, restrict CUDA to GPU 0, track PIDs/exit codes, and prevent concurrent project launches from taking the same GPU lock.

```bash
bash scripts/status.sh                    # latest run
bash scripts/status.sh RUN_ID
tail -f runs/RUN_ID/logs/pipeline.log
bash scripts/resume.sh bootstrap RUN_ID
bash scripts/stop.sh RUN_ID
```

Replace `RUN_ID` with the printed identifier. Stop signals only the tracked project process. Resume keeps the run ID/configuration/seed split, checks artifact hashes and JSON/NPZ/checkpoint integrity, and reruns incomplete or invalid stages. A stopped process is not counted as a successful run; inspect the persisted exit status and report.

## Profiles and actual human video

| Profile | Inputs and budget |
|---|---|
| `bootstrap` | Provided task seed; up to 20 target attempts, goal of 3 successful teachers, 300 BC steps, 3 held-out seeds. |
| `fast` | Provided task seed; 8 candidate source demos, up to 30 target attempts, goal of 10 successful teachers, 2,000 BC steps, 10 held-out seeds. |
| `full` | Actual human clips, CLIP ranking, then Qwen interpretation; bounded teacher/BC evaluation budgets. |

```bash
bash scripts/launch.sh fast

# Later, with an actual human can-to-bin clip:
cp /path/to/real_human_can_clip.mp4 data/human/
bash scripts/setup.sh full
bash scripts/launch.sh full
```

`setup.sh full` installs perception dependencies; model weights are loaded only when full inference runs. Full mode samples 4–6 ordered frames, computes CLIP similarity using `openai/clip-vit-base-patch32`, and interprets the selected clip with `Qwen/Qwen2.5-VL-3B-Instruct`. The models run sequentially. Raw Qwen text, sampled/annotated frames, parsed JSON, and semantic mapping assumptions are retained.

Without a real clip, full mode reports **`requires_human_video`**, skips task interpretation/retargeting/training, and produces an honest degraded report. Bootstrap explicitly reads `data/task_seeds/can_to_bin.json`; it never treats simulator video as human evidence. Unknown mass, friction, and contact state remain `unknown` in the semantic task. Metric target poses still come from the simulator. **Real CLIP/Qwen inference is untested in the recorded no-human-video runs. Optional SAM2 segmentation is not implemented.**

## Five stages and contracts

| Stage | Executed behavior | Main artifact |
|---|---|---|
| 1. Selection | Check actual HDF5 metadata/array lengths; deterministically select demos by length and diverse initial object positions. Full mode additionally ranks actual human clips with CLIP. | `01_selection/selection.json` |
| 2. Task | Validate the provided seed or Qwen-derived can-inside-bin semantics, retain unknown physics, and declare the semantic-to-simulator map. Unsupported goals are rejected. | `02_task/task_spec.json` |
| 3. Retargeting | Extract source object-relative grasp/lift geometry; execute a live scripted Panda controller; collect UR5e episodes through closed-loop task-space control. Preserve failed attempts. | `03_retarget/source_trace.npz`, `attempts.csv`, `rollouts/` |
| 4. Learning | Train a 2×128 MLP only on successfully executed UR5e demonstrations. Save model/optimizer/scaling state; load the saved model in a fresh process for closed-loop evaluation. | `04_policy/checkpoint.pt`, `train_log.csv`, `eval.csv` |
| 5. Diagnosis | Apply explicit simulator-state rules, preserve unsupported causes as unknown, request feedback, and attempt bounded additional collection/retraining with the same held-out evaluation seeds. | `05_diagnosis/failure_report.csv`, `feedback_requests.json`, `before_after.json` |

Source HDF5 provenance is public [robomimic Can / PH / v1.5](https://huggingface.co/datasets/robomimic/robomimic_datasets/tree/main/v1.5/can/ph). These are human-operated **Panda robot demonstrations**, not human activity videos. In this dataset's 14D `obs/object` array, the first seven values are EEF-relative pose; world can position is `7:10` and quaternion is `10:14`. The parser checks the actual schema rather than replaying old source joint commands on UR5e.

The shared online feature builder produces **26 dimensions**: EEF/object/goal XYZ (9), object-minus-EEF and goal-minus-object (6), gripper opening/contact/object lift (3), and an observable event-based phase encoding (8). Collection and evaluation use the same phase tracker and feature builder; phase features are not expert trajectory labels supplied at evaluation. Normalization is fitted only on training episodes, and train/evaluation seed overlap is rejected.

The learned intermediary is **4D**: three normalized XYZ deltas plus a gripper output. Training uses XYZ MSE and gripper BCE. The installed controller adapter maps this to the actual **7D `BASIC/OSC_POSE`** vector through `create_action_vector`: XYZ delta, three zero orientation deltas, and gripper command. One normalized XYZ unit corresponds to 0.05 m per control step on the tested controller; actions are clipped to `[-1,1]`, positive gripper closes, and the initial orientation is held. Panda and UR5e joint angles are never copied between embodiments.

Episode provenance keeps these distinct:

- `source_scripted`: the executable Panda source-policy smoke branch; it is scripted, not pretrained.
- `source_retargeted`: UR5e control uses geometry extracted from an identified source demonstration plus target simulator feedback.
- `target_scripted_bootstrap`: explicitly separate fallback teacher data if retargeting fails.
- `learned_policy`: actual saved-model closed-loop UR5e evaluation.

Task success is the simulator's actual placement/release predicate. Teacher success, genuine retargeting success, and learned-policy success are reported separately. Zero successful teachers blocks training; it never triggers random or invented data.

Diagnostic causes are deterministic rules on simulator state. In particular, the legacy `unreachable_waypoint` code means the controller did not reach its early waypoint before the horizon; it does not prove kinematic infeasibility. Inspect saved phases, contact, joint-limit margins, and state trajectories before assigning a physical cause.

## Results and evidence

Start with the [five-stage success videos and current issue specification](results/review_20261105/README.md). This review includes all nine successful fast-policy evaluations, a bootstrap failure-to-success feedback example, a same-seed teacher/policy failure comparison, and independently checked one-second post-release stability. These are frozen-model replays of the no-human-video path; full human-video transfer remains untested.

See [recorded experiment summaries](results/README.md). Raw local runs are under `runs/RUN_ID/`, including `FINAL_REPORT.md`, input hashes, environment/configuration, stage status/exception traces, executed trajectories, checkpoints, evaluation CSVs, and videos. Large datasets, models, human clips, and raw runs are gitignored.

An executed bootstrap run, `bootstrap_20261105T181834Z`, recorded:

| Measurement | Result |
|---|---:|
| Live scripted Panda | 1 / 1 successful |
| Source-retargeted UR5e teachers | 3 / 3 successful |
| Learned UR5e policy, held-out seeds 10000–10002 | 0 / 3 successful |
| Learned policy after 2 extra successful teachers and retraining, same seeds | 1 / 3 successful |
| Mean XY placement error before → after | 0.654 m → 0.259 m |

This is a small simulation baseline and a measured corrective iteration, not a generalization guarantee. Later runs and their exact settings belong in the results index. Comparison videos label available Panda/UR5e teacher/learned-policy panels by provenance and identify missing human input.

## Development and verification

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python -m cross_robot_real2sim.cli check
python -m pytest -q
python -m cross_robot_real2sim.cli report --run-id RUN_ID
```

The CLI also exposes `run --config ... [--resume]` and `stage --step 1..5 --config ... --run-id RUN_ID` for controlled debugging. Use the detached launch scripts for long jobs. Each stage's `run.py` is independently executable with `--config` and `--run-dir`; required upstream artifacts must exist.

Tests cover task/array contracts, real HDF5 layout, controller dimensions/clipping, online features, successful-teacher filtering, saved-policy loading, seed separation, goal predicates, diagnostic evidence, missing-human integrated behavior, and isolated detached-process lifecycle fixtures. Simulator tests are marked `sim`; process/integration tests are marked `integration`.

The source tree retains the five stage boundaries under `src/cross_robot_real2sim/`, with shared `adapters/`, `common/`, `orchestrator/`, and `viz/` modules. Small YAML profiles control attempt caps, training steps, horizon, feedback budget, and disk/runtime limits.
