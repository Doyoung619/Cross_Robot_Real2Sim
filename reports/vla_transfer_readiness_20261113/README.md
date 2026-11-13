# Pretrained VLA cross-robot transfer: readiness audit

**Executed follow-up:** the [minimal source-only VLA pilot](../../results/vla_minimal_20261113T185314Z/README.md) now implements a bounded version of B-aware selection, static conversion and LoRA training. B task success was not achieved (0/2 before and after). This document retains the earlier preparation snapshot; use the linked measured report for current execution status.

For the fastest proposed source-only execution experiment, see the
[plate-pushing pilot and conditional timing estimate](SOURCE_ONLY_PILOT.md).
Its additional original A dataset is downloaded; its conversion/training/eval
pipeline is not yet implemented. The bowl task below records the earlier audit.

Prepared on November 13, 2026. This report addresses the requested experiment:
retain an already trained Robot A VLA, select relevant existing A demonstrations
(or future human videos), process them offline using B's known embodiment, and
fine-tune adapters in that same VLA for B. **Do not collect B execution
demonstrations, including simulator rollouts, for training.** B execution is
reserved for final evaluation. This supersedes the earlier proposal to collect
twenty successful B training rollouts. That proposal did not match the request.
The historical state-MLP baseline does not establish this result.

## Selected minimum experiment

- A: Franka Panda, seven arm joints. B: UR5e, six arm joints.
- Task: **pick up the black bowl from table center and place it on the plate**.
  One pickup and placement; no drawer opening or multi-object sequence.
- Source policy: [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero),
  pinned revision `a24f1f380461b665158d52acd140152c79e8b36d`.
  Its saved training configuration identifies `lerobot/libero` as its training
  dataset. The downloaded policy contains 450,046,176 parameters.
- This is a credible pretrained source candidate. **Its closed-loop success on
  the selected task has not yet been measured locally.** Successful human
  demonstrations are not evidence that this checkpoint succeeds at deployment.
  Source-policy evaluation is a required gate before claiming transfer.

## Downloaded and verified materials

The pinned file inventory is [download_manifest.json](download_manifest.json).
It records actual local paths, revision, bytes, elapsed download time and SHA256
for every file. All 610 files downloaded without error: 1,923,649,618 bytes.

| Material | Actual state |
|---|---|
| Complete pretrained VLA weights | Downloaded, 906,712,520 bytes; strict model load passed |
| VLA configuration, preprocessing, normalization | Downloaded |
| Backbone tokenizer and processor metadata | Downloaded; separate backbone weights unnecessary for loading complete VLA |
| Original single-task LIBERO HDF5 | Downloaded, 589,630,943 bytes |
| LIBERO scenes/object assets | Downloaded, approximately 422 MB |
| LeRobot training source | Downloaded/extracted version 0.4.3; imports tested |
| LoRA/Transformers supporting packages | Installed in isolated `runs/vla_readiness_20261113/deps` overlay |
| LIBERO environment code | Downloaded `hf-libero` 0.1.3 source archive |
| Compatible robosuite code | Downloaded 1.4.0 wheel; not installed into existing environment |
| Existing UR5e model | Available in existing robosuite environment |
| B-conditioned offline source processing | Not implemented; geometry-only source selection is provisional |
| B training execution demonstrations | **Prohibited by experiment protocol**, not a missing collection task |
| Offline kinematic pseudo-labels derived from source | Not generated; hypothetical labels would not prove B task success |
| B VLA checkpoint and held-out success | Not trained/measured |

The existing project uses Python 3.10, PyTorch 2.6.0+cu124 and robosuite 1.5.2.
Current LeRobot main requires Python >=3.12, so the tested model-only overlay
uses 0.4.3. LIBERO requires robosuite 1.4.0; install that in a separate simulation
environment instead of replacing the working project's 1.5.2 stack. A model
import/gradient test does not certify that the complete LeRobot training CLI or
the LIBERO simulator is ready.

## Actual source selection

The task-specific original file has **50 demonstrations and 5,882 steps**.
All 50 have consistent observation/action lengths and positive terminal rewards.
All 50 were inspected. Select the shortest successful coherent demonstration,
then repeatedly choose the start bowl position farthest from the selected set
(max-min Euclidean distance). Joint addresses are derived from each saved XML;
the bowl position is read from its free-joint state, not guessed from EEF data.

Selected ten: `demo_43`, `demo_12`, `demo_33`, `demo_22`, `demo_30`, `demo_13`,
`demo_14`, `demo_6`, `demo_10`, `demo_47`: **1,141 source steps**.
See [readiness.json](readiness.json) for per-episode positions and counts.
This is an initial geometry-diversity baseline. It does not yet select by B's
reachability, collision constraints or gripper compatibility. The ten IDs are
provisional until that offline analysis is implemented. Do not treat these ten
A episodes as executed B demonstrations.

## Required conversion and adaptation

1. Run the downloaded A policy in the matching Panda/LIBERO scene; measure ten
   fixed-start closed-loop episodes, verify camera mappings and normalizers.
   Reconsider the checkpoint/task if it does not meet the source success gate.
2. Select **existing source episodes/segments** using task relevance and source
   quality, followed by B's offline kinematic feasibility, joint limits,
   collision clearance, gripper aperture/orientation and motion coverage.
   Use B's known model, not B rollout success, to make these decisions.
3. Extract source object-relative EEF poses, object motion, gripper/contact
   intent and timestamps. Use calibrated frame conversion, offline IK/FK and
   continuity/geometry checks to construct hypothetical B states/action labels.
   These are **source-derived kinematic pseudo-labels**, not observed successful
   B trajectories. Do not run B physics to collect training pairs or filter by
   B task reward. Log all parent source IDs, conversions and uncertainties.
4. Preserve the complete pretrained VLA weights. Freeze the visual/language
   backbone and inject rank-8 LoRA into action-expert Q/V projections and action
   input/output projections. Initially train 5,000 updates, batch eight, LR
   1e-4, action chunks of fifty. Train on source-derived processed pairs, keeping
   source images explicitly identified as source images. A paired A image and
   hypothetical B state/action require a deliberate shared observation
   representation; do not pretend they are native B camera observations.
   This representation and any input/output adapters remain design work.
5. Choose hyperparameters and checkpoints using held-out **source** episodes
   and offline kinematic checks. Freeze preprocessing, data selection and
   weights before B evaluation. Compare unadapted and adapted VLA on ten held-out
   B starts. Do not feed target rollout actions, rewards or failure traces back
   into training or selection under this source-only protocol. Preserve failures.

The central research uncertainty is observation transfer: B's arm and wrist
camera differ from A's. Masking the arm, object-centric views or a shared
geometric representation are possible design choices, not demonstrated fixes.
Offline IK can check pose feasibility but does not establish successful grasps,
friction/contact behavior or task completion. Future human video additionally
requires uncertain 3D object/hand reconstruction; it does not directly contain
robot action labels. This path is not implemented or downloaded by this audit.

Both controllers expose Cartesian 7D actions (translation 3, rotation 3,
gripper 1), despite their different arm joint counts. Coordinate frames,
orientation units, gripper conventions and camera calibration still require
validation; equal dimensions do not establish compatibility. Original HDF5 is
20 Hz with 128px images, while the training-format dataset reports 10 Hz and
256px images. Resampling and image conventions are implementation work, not a
file rename. The legacy Can-to-bin code does not implement this bowl task.

## Measured timing and estimate limits

Hardware: RTX 4090, 24 GB. Other GPU work was active; none was stopped.
Strict loading of the real source policy and finite-loss forward/backward/AdamW
updates succeeded. Rank-eight LoRA adds **334,592 trainable parameters**.
The base checkpoint was loaded before adapter injection.

| Measurement | Batch 1 | Batch 8 |
|---|---:|---:|
| Warmup updates | 2 | 2 |
| Timed updates | 6 | 6 |
| Median seconds/update | 0.0574 | 0.08915 |
| Peak PyTorch allocated GPU memory | 1.05 GiB | 2.28 GiB |

Evidence: [benchmark.json](benchmark.json),
[benchmark_batch8.json](benchmark_batch8.json).
Each benchmark repeats one real **Panda** batch. It excludes data loading,
augmentation, checkpoint saves and simulation. It does not measure source task
success, target training quality, adaptation convergence or full-system memory.
These are model-only lower bounds, not end-to-end training promises.

At the measured batch-eight rate, 5,000 updates require about **7.4 minutes** of
model updates alone; 20,000 require **29.7 minutes**. These measurements remain
useful as rough adapter-update throughput. The revised observation/action
adapters and actual preprocessing may change the timings.

The previous **6–16 hour** total estimate assumed executed B teacher data and
is withdrawn for the corrected source-only design. B-aware selection, offline
pseudo-label generation and observation transfer have not been benchmarked.
The following training-only planning ranges are conditional, not a completion
promise for the source-only research pipeline:

| Remaining work | Estimate, conditional on no major integration failure |
|---|---|
| First 5,000-update adapter training, with I/O/checkpoints | 15–45 minutes |
| Longer 20,000-update run, if useful | 45–120 minutes |
| B-aware selection and offline processing | Not estimated from execution evidence yet |
| Observation-transfer integration and final B evaluation | Not estimated from execution evidence yet |

The remaining work is source-to-B representation and offline processing,
followed by final evaluation. No additional executed B training data will be
collected. The ten-minute audit/preparation does not constitute a completed
source-only transfer, and the measured source gradient tests do not establish
that it will succeed.

## Reproduce the completed preparation

Run from the repository root:

```bash
.venv/bin/python scripts/prepare_vla_transfer.py
.venv/bin/python scripts/audit_vla_materials.py
.venv/bin/python scripts/benchmark_vla_adapter.py --batch-size 8
```

The last command needs the isolated dependencies listed in
[isolated-overlay.lock](isolated-overlay.lock), in addition to the existing
project environment. Downloaded simulation archives and hashes are listed in
[package_archives.json](package_archives.json). The experiment design is
[vla_transfer_minimal.yaml](../../configs/vla_transfer_minimal.yaml); it is
explicitly not a runnable configuration for the historical MLP pipeline.

Official references:
[VLA checkpoint](https://huggingface.co/lerobot/smolvla_libero),
[its training configuration](https://huggingface.co/lerobot/smolvla_libero/blob/a24f1f380461b665158d52acd140152c79e8b36d/train_config.json),
[source training dataset](https://huggingface.co/datasets/lerobot/libero),
[original LIBERO data](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets),
[official LoRA workflow](https://huggingface.co/docs/lerobot/peft_training).
