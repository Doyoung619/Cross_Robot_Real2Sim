# Coding Agent Execution Prompt — Cross-Robot Real2Sim MVP (1× RTX 4090)

You are the lead robotics engineer and autonomous coding agent for a research prototype. **Do not merely propose a design. Implement, install, run, debug, test, and report a real end-to-end MVP now** in the current workspace. Work independently within this repository. Avoid asking me questions unless a genuine permission/credential barrier prevents further work.

## 0. Mission, research context, and constraints

Professor's priority is: **get the entire 5-stage pipeline working first, diagnose actual bottlenecks, and only then choose which steps deserve novel research**. A future paper may present both the integrated framework and nontrivial improvements to specific stages.

Five stages (retain these exact boundaries):
1. Task-aware selection of human videos and source robot demonstrations/policies.
2. Extraction of task goals and constraints from the selected human video(s).
3. Extraction/generation of source-grounded robot trajectories and cross-embodiment retargeting.
4. Learning/optimizing a target robot policy using executed synthetic target trajectories.
5. Detecting failed/suboptimal behavior and unknowns; produce a feedback request and test one corrective iteration.

**Hardware:** single NVIDIA RTX 4090 (24 GB VRAM); headless Ubuntu/Linux over SSH. The pipeline **must continue after SSH/VS Code disconnect**. Never rely on a foreground SSH session for long-running training or rollout jobs. Build detached `tmux` launch/status/resume/stop scripts with logs, timestamps, PID/session tracking, checkpointing, and exit codes. `tmux` persistence covers SSH disconnects, NOT necessarily host reboot. Use only GPU 0 and **do not run heavyweight models simultaneously**.

**Speed over scale:** first execute a tiny, valid, traceable pipeline; then run slightly larger experiments. Do not start with Isaac Sim, huge VLA training, ROS, privileged cloud APIs, paid APIs, sprawling data downloads, or high-resolution RGB learning. Use public/open models only. Do not make destructive changes outside this repository; avoid `sudo` and changes to the user's shell configuration.

**Core scenario:** `Can` pick → lift → transport → place. Robot A = Franka Panda (7 DoF), robot B = UR5e (6 DoF), simulator = robosuite/MuJoCo `PickPlaceCan`. Source data = publicly available robomimic `Can / PH / v1.5 / low-dimensional` HDF5. The target must learn from executed synthetic robot-B demonstrations rather than human teleoperation of B. Source provenance and teacher type must be recorded. Initial evaluation may use privileged sim object poses, explicitly flagged `oracle_sim_state`; never claim vision-only or real-world transfer in that mode.

Input video constraint: a real human `.mp4` may not exist on the server. **Do not block the initial end-to-end run on this.** Provide two explicitly distinct modes:
- `bootstrap`: all five modules execute with a **provided/configured task spec** rather than pretending an invented/simulated clip is human video; Step 1 uses metadata/baseline selection. If needed, generated robot simulation frames may be used only for model smoke checks, clearly labeled `synthetic_robot_video`, never `human_video`. Produce an end-to-end test, including Step 5.
- `full`: when the user places at least one real human clip in `data/human/`, actually run CLIP and Qwen on sampled video frames, get a task JSON, and connect this JSON to retargeting. Add SAM2 tiny segmentation if practical, but do not allow an optional segmentation dependency to block the main pipeline. A missing human video must yield `requires_human_video`, not `success` for the human-video branch.

Aim to complete `bootstrap` ASAP; then prepare and, if real video files exist, execute `full`. **Never fake successes, train on random noise, create canned artifacts, report success from a hard-coded bool, or silently substitute an unrelated task.** Legitimate scripted/heuristic fallbacks are allowed only if marked clearly in provenance and metrics.

## 1. Public resources — verify compatibility before installing

- Source dataset (only this file initially, about 46.9 MB):
  `https://huggingface.co/datasets/robomimic/robomimic_datasets/tree/main/v1.5/can/ph`
  `v1.5/can/ph/low_dim_v15.hdf5`
  Download with `huggingface_hub.hf_hub_download(..., repo_type='dataset')` or `hf download robomimic/robomimic_datasets v1.5/can/ph/low_dim_v15.hdf5 --repo-type dataset --local-dir data/source`. Do **not** accidentally download the entire dataset repository. Inspect actual HDF5 keys/metadata/array shapes before choosing an extractor. Do not assume old robomimic checkpoint compatibility with current robosuite.
- Robot sim: `https://robosuite.ai/` / `https://robosuite.ai/docs/modules/environments.html` / `https://robosuite.ai/docs/modules/controllers.html`. API/version details (composite controller, controller config, gripper action, observation keys, `action_dim`) MUST be discovered and tested against the installed version. Robosuite controller action is NOT necessarily `[dx,dy,dz,gripper]`; implement an action adapter.
- Step 1 visual text similarity: `openai/clip-vit-base-patch32` via Transformers, after a deterministic metadata selector.
- Step 2 compact VLM: `Qwen/Qwen2.5-VL-3B-Instruct`, torch inference (bf16/fp16), ordered 4–6 video frames at reduced resolution. Follow the installed Transformers/model-card APIs, not guessed Python classes. Output structured, validated JSON. Optional segmentation: SAM2.1 Hiera Tiny from `facebookresearch/sam2`, with optional box/point initialization; contact inference from 2D is only a hypothesis, not a proven physical contact.
- Step 3: source object-relative 3D waypoint extraction plus the target's own robosuite OSC/IK/closed-loop feedback control; never copy source joint angles into UR5e.
- Step 4: small PyTorch MLP Behavior Cloning first (e.g. 2×128 hidden), not ACT/SmolVLA initially. Learn task-space actions, convert through target action adapter.
- Step 5: deterministic simulator-ground-truth diagnostic rules (no large model), feedback manifest, one extra data-collection/retrain/evaluation iteration.
- Optional later comparison only: LeRobot ACT or SmolVLA. Do not put these on the critical path.

## 2. Exact target repository architecture

Create this modular, editable structure (extra files are fine, but retain the five stage boundaries and interfaces). **Treat the current repository as the root; if the working directory is already named `cross_robot_real2sim`, do not create an accidentally double-nested repository**:

```text
cross_robot_real2sim/
├── README.md
├── pyproject.toml
├── requirements-sim.txt
├── requirements-perception.txt
├── configs/
│   ├── bootstrap.yaml                # minimal no-human-video test
│   ├── fast.yaml                     # 4090 practical small run
│   └── full.yaml                     # actual human-video processing
├── data/
│   ├── human/README.md               # where and how to place .mp4 clips
│   ├── source/                       # robomimic Can PH HDF5
│   └── task_seeds/can_to_bin.json    # explicitly manual/oracle task seed
├── src/cross_robot_real2sim/
│   ├── __init__.py
│   ├── cli.py                        # run/stage/check/report CLI
│   ├── schemas.py                    # typed JSON/NPZ contracts
│   ├── common/{config.py,logging.py,io.py,provenance.py,seeding.py}
│   ├── adapters/{robosuite_env.py,robot_action_adapter.py,source_hdf5.py}
│   ├── stage1_selection/{run.py,metadata_baseline.py,clip_ranker.py}
│   ├── stage2_task/{run.py,seed_baseline.py,qwen_vl.py,sam2_optional.py}
│   ├── stage3_retarget/{run.py,source_trace.py,object_frame.py,waypoints.py,controller.py,rollouts.py}
│   ├── stage4_policy/{run.py,build_dataset.py,bc_model.py,train.py,evaluate.py}
│   ├── stage5_diagnosis/{run.py,detectors.py,feedback.py,retrain_check.py}
│   ├── orchestrator/{run_pipeline.py,cache.py,run_state.py}
│   └── viz/{render_rollouts.py,comparison_video.py,plot_metrics.py}
├── scripts/
│   ├── setup.sh
│   ├── preflight.sh
│   ├── launch.sh                    # tmux: detached durable entry point
│   ├── status.sh                    # status + GPU + log tail
│   ├── resume.sh                    # rerun incomplete/failed stages
│   ├── stop.sh                      # graceful stop, no broad killall
│   └── run_foreground.sh            # also used inside tmux
├── tests/
│   ├── test_stage_contracts.py
│   ├── test_sim_smoke.py
│   ├── test_dataset_parser.py
│   ├── test_action_adapter.py
│   ├── test_goal_evaluation.py
│   └── test_end_to_end_bootstrap.py
└── runs/                           # gitignored, one directory per run ID
    └── <run_id>/
        ├── manifest.json
        ├── environment.json
        ├── status.json
        ├── logs/pipeline.log
        ├── 01_selection/{selection.json,status.json}
        ├── 02_task/{task_spec.json,annotated_frames/,status.json}
        ├── 03_retarget/{source_trace.npz,attempts.csv,rollouts/,stats.json,status.json}
        ├── 04_policy/{train_data.npz,checkpoint.pt,train_log.csv,eval.csv,status.json}
        ├── 05_diagnosis/{failure_report.csv,feedback_requests.json,before_after.json,status.json}
        └── videos/{source.mp4,target_teacher.mp4,target_bc.mp4,comparison.mp4}
```

Avoid creating placeholder files and calling them results. Every stage must have independently runnable `run.py` logic AND integrate through one CLI:

```bash
python -m cross_robot_real2sim.cli check
python -m cross_robot_real2sim.cli run --config configs/bootstrap.yaml
python -m cross_robot_real2sim.cli run --config configs/fast.yaml --resume
python -m cross_robot_real2sim.cli stage --step 3 --config configs/fast.yaml
python -m cross_robot_real2sim.cli report --run-id <run_id>
```

Contract rules: meters, seconds, radians; named frames (`source_world`, `target_world`, `object_relative`, `human_image_px`); timestamps and array shapes; identical robot adapters across train and rollout; JSON for lightweight metadata, NPZ for trajectories, CSV for evaluation, MP4 for evidence. Every output contains input/source IDs, schema version, seed, mode, model/controller name, and whether privileged state was used. Explicit `status`: `completed`, `completed_degraded`, `failed`, `skipped_missing_input`; record exceptions and reasons. Do not claim `completed` for perception if no human video exists.

## 3. Implementation of the five stages (be specific)

### STEP 1 — Data Selection
- Input: task text (`place the can inside the bin`), human video manifest (may be empty in bootstrap), HDF5 demo IDs, robot metadata, target task spec.
- Implement metadata filter and list of source HDF5 episodes with consistent sizes and provenance, choose small deterministic sample.
- In `full`, sample up to 4–6 frames from each real clip and compute CLIP text/image similarity, log all scores, ranking and distractor rejection if a distractor exists. Do NOT invent a distractor. For source demo selection, use available task metadata and diversity rather than treating HDF5 as human RGB video.
- Output `selection.json`: selected human clip paths, selected source demonstration IDs, numerical scores where actually computed, source IDs, reasons, selection strategy.
- `bootstrap`: real, functioning metadata selector; explicit `human_clip_selection: skipped_missing_input` if no video. This is not a CLIP achievement.

### STEP 2 — Task Goal and Constraints
- Input: selected clips/task prompt and simulation task definitions.
- `bootstrap`: read `data/task_seeds/can_to_bin.json`, validate via Pydantic/dataclass schema, set `task_source='provided_seed'`, and generate a valid task spec. Preserve unknown mass/friction/contact state as `unknown`. This is not a vision result.
- `full`: run Qwen2.5-VL-3B-Instruct on a few ordered frames (sequential on GPU), ask for JSON with object=`can`, goal relation=`inside`, goal receptacle=`bin`, stages/constraints and unknowns. Parse/validate and preserve raw model text for debugging. Implement an explicit map from video semantic labels to simulator object IDs, with declared assumptions. Optional SAM2 tiny: produce visual masks/annotated frames if correctly initialized; do not let it be a blocking dependency for a first working result.
- Do not hallucinate metric 3D hand trajectories, exact object masses, or contact from monocular RGB. For bootstrap target poses use simulator state; label `goal_pose_source=simulator_oracle`.
- Output `task_spec.json`, annotated frames if available, and detailed provenance.

### STEP 3 — Source-Grounded Retargeting + Target Synthetic Demos (critical path)
- Inspect downloaded Can HDF5 first. Report keys and actual shapes. Extract EEF trajectory, source object-relative geometry, possible grasp/release segments, or state clearly which fields are unavailable. Use source demonstration timestamps and source pose orientation if available.
- Initialize both Panda and UR5e in `PickPlaceCan` headlessly. Verify controller, gripper and action dimensions with the installed robosuite version. If actions use a 6D pose delta + gripper command, provide the correct mapping, orientation hold, clipping and validation; do not concatenate 4 values and assume the environment accepts them. Check `print_action_info`, `create_action_vector`, and compatible controller configuration if applicable to the installed version.
- Convert the source to a small number of **object-relative 3D waypoints** in the target scene: pre-grasp → grasp → close → lift → transport → lower → open → retreat. Implement closed-loop waypoint tracking and event-triggered gripper actions. Add reachability/collision/gripper checks; instrument each failure.
- Run a **scripted Panda policy** at least once from live simulator feedback, so the source-policy branch is executable and not only a dataset replay. Label this `source_policy_type=scripted`; do not call it pretrained/learned.
- Generate at least one **genuine source-grounded Panda→UR5e rollout** and attempt target can placement. Store failed trials as valuable evidence. Try small, bounded sets of starting positions and grasp parameters rather than infinite optimization. Use a configurable attempt cap.
- A separate **target-oracle scripted waypoint controller** is allowed ONLY as a debugging/bootstrapping fallback if actual source-grounded retargeting cannot succeed yet. Label each episode `teacher_origin=source_retargeted` or `teacher_origin=target_scripted_bootstrap`; NEVER mix those labels in reported transfer success. Retargeting success requires actual sim object placement, not plausible video alone.
- Outputs: source trace NPZ, target rollout NPZ, attempt-level CSV with failure codes, and at least one source/target rendered video if rendering works.
- If no transfer success: retain all failed retargeting attempts and allow the downstream training/testing on **separately marked scripted teacher data**. Final report must say `end-to-end infrastructure working; genuine cross-robot transfer NOT YET verified`.

### STEP 4 — Target Policy Learning + Evaluation
- Train a lightweight MLP BC policy from target UR5e **successfully executed** rollouts generated in Step 3. Minimum: state features include robot EEF position, object position, target position, gripper status; optional phase features MUST be computed online at evaluation from observable signals, not from expert-only hindsight labels. Use a consistent feature builder at collection and inference. Keep feature dimensions as runtime-validated config; do not quietly change a declared 16D schema.
- Task-space action may be `[dx,dy,dz,gripper_logit]` (4D intermediary), but the robot adapter must map it to the installed simulator's actual action dimensions/orientation representation. Explicitly define delta units, normalization, and clipping. Train with MSE for XYZ, BCE for gripper; normalize using training data only. Save `checkpoint.pt`, optimizer state, scaler/normalization stats, seed, model config.
- For `bootstrap`: very small dataset (3–5 successful target rollouts if possible), short training budget, 3–5 fresh evaluation seeds. For `fast`: 10–20 target teacher successes if obtainable, ~2k training steps, 10 held-out seeds; stop on an explicit compute budget, record achieved count if less. Teacher rollouts and learned policy must be reported separately.
- Run actual closed-loop UR5e episodes using the saved learned model loaded in a fresh process; save `eval.csv`, metrics and videos. An evaluation that executes but fails task is valid, truthfully reported as 0% success; do not fabricate positive results. If there are zero successful teacher episodes, mark training `blocked_insufficient_teacher_data` and proceed to Step 5 with diagnosis rather than inventing data.

### STEP 5 — Failure Attribution + Feedback Cycle
- Analyze all Stage 3 and Stage 4 actual rollouts with explicit goal predicate, distance-to-goal, lift achieved, object drop, premature release, unreachable waypoint, gripper action mismatch, collision when observable, horizon timeout, and goal parsing/source selection mismatch where supported. Keep ambiguous failures marked `unknown` rather than forcing an unsupported cause.
- Produce row-per-attempt `failure_report.csv` with step attribution, reason code, evidence, seed, teacher_origin, confidence type. Distinguish `diagnostic_oracle_used=true` from deployed observations.
- Produce `feedback_requests.json`: e.g. request increased grasp waypoint clearance, more target teacher starts near an observed failure, or task JSON correction. Implement one conservative, automatically selected intervention **if feasible**, collect limited additional trajectories, retrain or re-evaluate, and compare before/after on the **same held-out seeds**, plus log an additional seed set if available.
- Do not claim improvement merely because more data was collected; provide actual success/placement-error numbers. If no feasible intervention, log `intervention_not_executed` and why.

## 4. Resource budgets and run modes

Create YAML config profiles and enforce limits:

```yaml
# bootstrap.yaml — example values, adjust after smoke tests
mode: bootstrap
seed: 42
device: cuda:0
source_robot: Panda
target_robot: UR5e
environment: PickPlaceCan
human_video_required: false
stage2_backend: provided_task_seed
source_dataset: v1.5/can/ph/low_dim_v15.hdf5
source_demo_limit: 3
selected_source_demo_limit: 2
target_teacher_success_goal: 3
target_attempt_cap: 20
held_out_eval_seeds: 3
bc_train_steps: 300
render_width: 224
render_height: 224
control_hz: 20
horizon: 300
use_oracle_sim_object_state: true
optional_sam2: false
optional_vla: false
```

`fast.yaml`: 5–10 source demos, 10–20 target successes if possible, at most ~2k BC steps, at most 10 held-out seeds, small RGB frames. `full.yaml`: same compute discipline but enables real human clip selection and Qwen video interpretation; do not implicitly enable ACT/SmolVLA.

Do not use GPU for non-GPU tasks; Qwen, CLIP, optional SAM2 and BC should run **sequentially**, releasing model references and GPU memory between stages. Use inference mode/autocast when compatible. Verify `nvidia-smi`, driver and `torch.cuda.is_available()` before using CUDA. Cap disk/storage and input downloads; don't re-download files that exist and pass basic checks.

## 5. Headless deployment + SSH disconnection survival

Implement scripts so the user runs only these commands (from repo root):

```bash
bash scripts/setup.sh
bash scripts/preflight.sh
bash scripts/launch.sh bootstrap
# SSH may safely disconnect after launch returns successfully.
bash scripts/status.sh
bash scripts/resume.sh bootstrap
# Real human input later:
# cp /path/to/real_human_can_clip.mp4 data/human/
# bash scripts/setup.sh full
# bash scripts/launch.sh full
```

Implementation requirements for scripts:
- `setup.sh`: isolated virtual environment (`.venv` or conda only if already available), no sudo, **bootstrap installation installs only simulator + low-dimensional BC dependencies**; `bash scripts/setup.sh full` additionally installs perception dependencies on demand. `pip freeze` or lockfile for exact working versions, download only minimal Can HDF5; do not download multi-GB Qwen weights during the bootstrap setup. Fail clearly on broken packages. Avoid dependency conflicts by separating VLM environment if needed. Print commands to reproduce.
- `preflight.sh`: GPU memory, driver/CUDA/Python, disk available, dataset path/size/keys, minimal 3-step Panda and UR5e simulator smoke tests, offscreen render 3 frames, logging of package versions. Set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` appropriately for headless use; if EGL fails diagnose and try supported fallback without pretending hardware rendering worked.
- `launch.sh`: use `tmux new-session -d -s crossrobot_<run_id> ...` with **absolute repo paths**, initialize the venv INSIDE detached session, set `CUDA_VISIBLE_DEVICES=0`, `MUJOCO_GL=egl`, `PYTHONUNBUFFERED=1`, and run the CLI. Echo run ID, session, log path, and status command immediately. If `tmux` unavailable, a safe `nohup`+`setsid` fallback is acceptable, but store PID carefully. Do not start duplicate runs against the same run ID and GPU unintentionally.
- `run_foreground.sh`: `set -o pipefail`, stream timestamped logs to `runs/<run_id>/logs/pipeline.log` via `tee`, write a reliable final `status.json` containing overall exit code and stage outcomes even on error; handle SIGINT/SIGTERM gracefully. Ensure working directory and venv are correct even after SSH closes. Do not let `tee` hide errors.
- `status.sh`: display live/stopped state based on tmux/PID AND persisted final exit status; latest stage/status, tail ~30 log lines, `nvidia-smi` summary; process disappearance alone does not mean success.
- `resume.sh`: re-execute only missing/failed outputs after schema/file integrity checks, stable run ID, no fresh seed split unless explicit; must not skip a stage just because a stale file exists.
- `stop.sh`: graceful stop of only this project's tracked tmux session/PID (not `killall python`), leave artifacts/resume state intact.
- If one component fails, store exception and continue to diagnosis/report where scientifically valid; distinguish `partial` and `failed` overall states from completed transfer.

## 6. Integration and correctness gates (execute them, do not just write tests)

G0: installed environment, 3-step Panda+UR5e headless simulation and HDF5 schema inspection succeed.
G1: source-grounded Panda→UR5e trajectory conversion is attempted, producing actual rollout/state logs; if any succeeds, show actual placement success. Also show source scripted-policy execution.
G2: `bootstrap` Step 1→2→3 passes schema contracts and honest provenance, even with no human video.
G3: at least one UR5e teacher success (prefer source-retargeted, otherwise explicitly target-scripted fallback), and BC model trains, loads, and performs fresh held-out closed-loop evaluation. If impossible, block honestly and record why.
G4: Step 5 generates a real failure report from rollouts and feedback manifest, and performs a bounded follow-up iteration if possible.
G5: when a real human clip exists, `full` runs real CLIP and Qwen inference and preserves parsed-goal evidence. Otherwise document it as untested and ready for clip input, NOT done.

Write tests for schema validation, frame transforms, controller output dimensions and clipping, sim scene reset/seed determinism, goal success predicate, no train/eval seed overlap, correct policy load, missing human-video behavior, and pipeline artifact integrity. Make the tests runnable on CPU where possible; tag GPU/simulator integration tests.

**Primary output**: a short `runs/<run_id>/FINAL_REPORT.md` with:
1. exact commands/environment/models/dataset used;
2. table of Step 1–5 actual outcomes (`completed/degraded/blocked/failed`), input provenance and timings;
3. source Panda success, target scripted success, *genuine retargeting* success, BC success on unseen seeds, and sample counts (NEVER conflate them);
4. ranked top 3 real bottlenecks with evidence, failure examples, and next experiments;
5. links/paths to videos and artifact files;
6. anything still NOT implemented/tested, including whether real human video was used.

Also create `runs/<run_id>/videos/comparison.mp4` containing human clip if provided, Panda source rollout if available, UR5e retargeted or correctly labeled scripted controller rollout, and UR5e learned policy rollout. Preserve aspect ratios and annotate each panel with its provenance. If an input video does not exist, do not fake that panel; label it `NO HUMAN VIDEO SUPPLIED` or omit it with explanation.

## 7. Execution order and reporting policy

Execute this order, prioritizing code that *actually runs*:
1. Inspect machine and existing project, avoid overwriting user work. Create repo skeleton, environment/setup, HDF5 fetch/parser and robosuite smoke tests.
2. Prioritize Stage 3 target controlled pickup immediately (critical integration dependency); build source demo introspection and action adapter; test Panda and UR5e.
3. Implement all stage contracts and bootstrap Stage 1, 2, 3, 4, 5; execute the integrated bootstrap pipeline and fix errors.
4. Implement CLIP and Qwen real-video paths; optional SAM2 if it does not block pipeline. Run `full` only if actual human clip is present.
5. Launch the **real integrated run in detached tmux**, verify it is active or has an auditable completed exit status, inspect results, and print final concise report.

Do not stop after merely writing files. Continue through testing, runtime debugging and actual execution. Prefer a small honest working baseline to a larger fragile architecture. On a genuine unsolved robotics issue, record the precise failure and preserve a functional downstream debugging fallback; do not silently turn a failed transfer into a research success.

At the end respond concisely with:
- Which stages actually ran, which stage(s) failed, and why.
- The exact launch, status, resume, log-inspection and stop commands.
- The run ID and paths to the report/videos.
- Measured results and whether they are source-retargeted, scripted bootstrap, or learned-policy results.

**Start implementing and executing now.**
