# Fixed-GR1 cross-embodiment few-shot adaptation benchmark

The research hypothesis is **unverified**: under the same maximum number of target adaptation rollouts, failure-guided adaptation should yield higher independent task success. This repository implements a common-interface comparison and reports measured outcomes, including failures. It does not assert that hypothesis or call idea-level adaptations official paper reproductions.

## Common environment

- Source: Franka Panda, seven arm joints; the same 50 successful task-specific LIBERO demonstrations for every method.
- Target: the installed robosuite GR1 geometry and native seven-joint right arm, with a replacement Robotiq85. Floating base, torso, head, legs and left arm joints are removed after their fixed transforms are baked into the MJCF. The native right-arm joint limits are retained.
- Fixed setup: base `[-0.53, 0.23, 0.95]` metres; torso pitch `0.45` rad (25.8 degrees). No balance or locomotion controller is learned. Every task control step checks fixed-body position/quaternion drift against `1e-8`.
- MuJoCo / LIBERO-Goal, OSC_POSE at 20 Hz, task **Push the plate to the front of the stove**.
- Success: native `On(plate_1, main_table_stove_front_region)` for 20 consecutive control steps. Final-step native predicate and ever-success are also logged. A short smoke horizon is not a meaningful task-performance test.
- Policy: `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`; frozen base, rank-8/alpha-16 LoRA, 334,592 trainable parameters. Inputs: two RGB cameras, fixed instruction and 8D TCP/gripper state. Physical actions: 7D Cartesian pose/gripper commands; predict 50, execute five, ten flow-integration steps.
- Initial 100 optimizer updates, then 100 updates after each counted adaptation episode; batch four, AdamW learning rate `1e-4`, gradient clipping 1, bfloat16, native masked flow-matching loss. The smoke profile uses two updates solely to exercise code paths.

The earlier prototype's low source feasibility and failed target transfer are not erased. The benchmark again records every source probe and rejection. Its common preprocessor uses full-frame IK, shared camera calibration and future-validity masks for all methods, so anchor counts can differ from the earlier sparsely checked pilot. Acceptance remains 15 mm / 0.15 rad, native joint limits, the same static arm/environment penetration check, and at least 20 retained anchors. IK receives up to 160 iterations. These checks do not certify dynamics or all self-collisions.

## Independent state banks and protocol sealing

`prepare` creates disjoint **adaptation**, **development** and **evaluation** banks. Evaluation defaults to 50 unique valid starts; the full grid is `K = [0,1,2,4,8,16]`. State generation uses native LIBERO object/fixture placement seeds. It stores complete simulator state **and model body/site transforms**, not only object positions. All methods/checkpoints use exactly the same evaluation bank and fixed initial robot configuration. Plate positions are distinct at a 10 micrometre threshold, bank signatures are disjoint, already-successful starts and static penetrations are rejected. No target physics or success-search is used to construct the banks.

The native plate distribution is only 2 cm by 2 cm: `x=[.04,.06]`, `y=[-.03,-.01]`. These are genuinely distinct in-distribution starts, **not 50 repetitions of the old two scenes**, and not broad OOD generalization. Other objects and fixtures are sampled natively. Initial validity is not a guarantee of successful future task execution. Fifty starts do not establish unseen-task or real-robot transfer; exposure during original VLA pretraining is not audited.

All requested adaptation checkpoints are produced **before any final evaluation begins**. Checkpoints are the last weights after a predeclared update schedule; there is no performance-based checkpoint selection. Development rollout budget is explicitly zero. Code/configuration, camera/robot assets and state-bank hashes are recorded before training and checked before evaluation. Changes require a new run. Evaluation has a separate entry point and no path back to adaptation; only `role=adaptation` envelopes are accepted by the feedback API. This is an architectural separation, not an operating-system sandbox against malicious code.

## Physical interaction accounting

`rollouts.jsonl` appends a `started` event immediately before an episode's first `env.step`. A started episode counts once even if interrupted or it fails. Its completion records physical steps/outcome. Static IK, source replay, static rendering and state-bank construction count zero. No target teacher collector is present. A teacher collector added later must use this same counted entry point.

The ledger separates adaptation, development, final evaluation and smoke roles. Smoke experiment profiles still record their actual adaptation/evaluation roles; their profiles and results are separate from the performance pilot. Zero-interaction references actually use zero adaptation episodes. Other controls execute the same permitted adaptation starts but discard feedback for learning. No failed B action is an imitation target; all action labels originate in the A HDF5 and offline target transformations. Successful B actions are also not used in this A-only supervision track.

A checkpoint repeated under another method name/allowed budget can reuse evaluation **only if weights, inference seed, state bank and horizon are identical**. Such aliases reference the same trials, never extra independent observations. Tables distinguish allowed K, actual K, evaluated states and global physical counts. The curve's x coordinate is actual adaptation interactions. Frozen references are dotted horizontal references explicitly labelled as using zero interactions.

## Methods and feedback permissions

| ID | Implemented behaviour | B information used for learning |
|---|---|---|
| `pretrained` | Original VLA; no training | None; zero adaptation interactions |
| `random_lora` | Random three successful A episodes, static conversion, fixed-source LoRA | None; adaptation measurements discarded |
| `b_feasible_lora` | Four-pose B-feasibility filter, short/diverse-start selection, initial LoRA, then frozen | None; zero additional interactions |
| `fixed_source_train` | Same feasible initial data, matched additional LoRA updates | None; adaptation measurements discarded |
| `selection_only` | Failure geometry ranks unused A episodes; original static labels | Adaptation-only goal/contact summaries; no contact-offset correction |
| `retarget_only` | Fixed original A episodes; contact-offset search and new Cartesian labels | Adaptation-only goal/contact/response summaries; no new source selection |
| `our_full` | Failure-conditioned A retrieval, bounded static contact correction, relabeling, LoRA | Adaptation-only simulator diagnostics |
| `cei_inspired` | Sparse functional contact-patch alignment, sequential bounded optimisation, synthetic RGB/OSC labels | None; matched additional training on generated data |
| `strap_style` | Visual/state S-DTW subtrajectory retrieval and LoRA | Whole adaptation observation sequence; no failure-context filtering |
| `fert_inspired` | Preterminal failure-context retrieval and overlap-prioritized successful-A replay | Failure label plus adaptation visual/state context |

Every trainable method uses the same VLA, interface, optimizer-update ceiling, batch size and controller. Source/anchor counts may differ and are recorded. When no new source satisfies fixed feasibility, feedback methods log the reason and continue matched training on existing data; they do not relax joint/collision constraints or claim to have found new data. This matched-compute fallback differs from the earlier prototype's immediate stop. A runtime failure before a usable episode produces an explicit budget shortfall.

### Fidelity to the papers

**CEI-inspired** preserves point-direction functional similarity, the symmetric directional Chamfer expression and sequential target alignment. Our manually specified box-pad patches use bounded L-BFGS-B with numerical gradients. We replace original observation synthesis and policy/control with simulator RGB, SmolVLA/LoRA and Cartesian OSC labels, retaining a common TCP-feasibility gate. This is not the full point-cloud/joint-target CEI pipeline. [CEI paper](https://arxiv.org/html/2601.09163), [project](https://cross-embodiment-interface.github.io/).

**STRAP-style** preserves subtrajectory retrieval with S-DTW. It replaces DINOv2/CLIP and original policy training with the frozen SmolVLA vision tower and common LoRA. Initial queries come from synthetic target observations, and later queries from autonomous adaptation observations; successful target demonstrations are not supplied. The source pool is one task, not a multitask corpus. [STRAP paper](https://arxiv.org/abs/2412.15182), [official retrieval code](https://github.com/WEIRDLabUW/STRAP), [official policy code](https://github.com/WEIRDLabUW/robomimic_strap).

**FERT-inspired** preserves frozen visual/state retrieval, preterminal failure queries, one-or-two-frame S-DTW slope constraints and overlap-based priority updates with retention 0.88. It replaces successful target-fleet replay with successful A intervals converted offline. Our tower pooling/projection, 8D state, query time base and top-two retrieval differ from the original setup. No fleet system, original SFT/HG-DAgger initialization or successful-B replay buffer is reproduced. [FERT paper](https://arxiv.org/html/2610.25736), [project](https://cel-robot-learning.github.io/).

`cei_official`, `strap_official` and `fert_official` are registered as **unavailable**, with reasons. Selecting one produces an unavailable status, not a fabricated score. These inspired/style curves cannot establish superiority over faithful official reproductions.

## Source processing and diagnostics

Initial selection applies four-pose IK/static collision checks, then shortest and farthest initial plate starts. Full feedback ranks unused successful A segments using remaining push-direction alignment, plate displacement, rear approach and contact-window length. B diagnostics include native/stable success, final XY error, displacement, contact rate/force, surface gaps, action direction, object response, EEF controller-goal tracking error, collisions and joint-limit margins/violations.

Retargeting searches small contact offsets (at most 2 cm, vertical at most 1 cm), solves bounded IK and recomputes future Cartesian labels. Response-conditioned lookahead is 2–5 frames; CEI-inspired next-pose conversion uses one frame. These are heuristic controller inversions, not calibrated dynamics models. Objects follow A records during synthetic data generation. No B physics is stepped there. Feedback files link each counted episode to rankings, selected A intervals, offsets and added anchors. Masks exclude out-of-episode and statically invalid future targets.

Retrieval features use frozen VLA image tokens, coarse 4x4 pooling where possible, a fixed 64D projection and separately normalized 8D state. Source embeddings never use evaluation observations. Shared source inspection/encoding costs are recorded separately.

## Run commands

From `/home/dheo/Cross_Robot_Real2Sim` with the existing project `.venv`:

```bash
# List exact implementation/fidelity statuses.
.venv/bin/python -m cross_robot_real2sim.benchmark list

# Detached whole-run execution; survives SSH disconnection.
bash scripts/launch_cross_embodiment_benchmark.sh smoke
bash scripts/launch_cross_embodiment_benchmark.sh pilot
# Large single-training-seed matrix (can take many hours):
bash scripts/launch_cross_embodiment_benchmark.sh full

# Select only particular methods in a new detached run.
bash scripts/launch_cross_embodiment_benchmark.sh full benchmark_my_methods fixed_source_train our_full

# Explicit foreground prepare/run, e.g. a team's independent method experiment.
.venv/bin/python -m cross_robot_real2sim.benchmark prepare \
  --config configs/benchmark/full.json --run-dir runs/benchmark_team_run \
  --methods strap_style --budgets 0 1 2 --evaluation-states 50
.venv/bin/python -m cross_robot_real2sim.benchmark run --run-dir runs/benchmark_team_run

# Resume final evaluation of already-frozen checkpoints, or regenerate reports.
.venv/bin/python -m cross_robot_real2sim.benchmark evaluate --run-dir runs/RUN_ID
.venv/bin/python -m cross_robot_real2sim.benchmark report --run-dir runs/RUN_ID

# Read-only post-evaluation symptom, ablation and wall/CUDA cost summaries.
.venv/bin/python scripts/analyze_cross_embodiment_benchmark.py \
  --run-dir runs/RUN_ID --output-dir results/RUN_ID
```

Use a fresh run ID. Existing directories are refused. A supervisor enforces the profile deadline (smoke one hour, pilot two hours, full twelve hours), records failure/timeout, and exports any measured partial results. A GPU lease prevents overlapping benchmark supervisors; unrelated GPU processes are not stopped. Interrupted adaptation with already-counted interactions is **not silently replayed**: that run is preserved and a new run is required. Final evaluation resumes completed episode files without duplicating them. Bank hashes and checkpoint hashes must still match.

`full.json` includes all ten implementations and K=0/1/2/4/8/16. `pilot.json` is a limited 50-start comparison of pretrained, fixed-source and full-feedback methods at K=0/1. `smoke.json` uses two starts, a 20-step horizon and two updates: it validates execution, not task performance. To study training variation, create a new config with `training_seeds: [0,1,2]`; each seed is reported separately rather than pooling repeated starts as independent training evidence. Batch four and learning rate 1e-4 are the supported common learning settings.

### Local prerequisites

The benchmark reuses tracked helper functions from the earlier pipeline. Its GR1 runtime is a frozen compatibility snapshot so that unrelated uncommitted humanoid work need not be committed. Existing work is not overwritten.

Required local materials are under `runs/vla_readiness_20261113/`: `model/` (complete VLA weights/config/normalizers), `backbone/` (processor/tokenizer metadata), `deps/` (tested LeRobot0.4.3 overlay), `sim_deps/` (robosuite1.4), `hf_libero-0.1.3/`, `assets/`, and `dataset/libero_goal/push_the_plate_to_the_front_of_the_stove_demo.hdf5`. The base `.venv` provides installed robosuite1.5.2 GR1 mesh assets and CUDA PyTorch. The original material-preparation scripts and manifests document downloads. Model/data hashes are recorded in each protocol; no source HDF5, base weights, optimizer or synthetic image arrays are uploaded by this benchmark.

The plate HDF5 source is [LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets), pinned material revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`, file SHA256 `36b4e1bced49d2f4ff6b2fce6b1596a63978e14199e2513cd0df71e127bf47a6`. Cross-machine installation of the complete earlier stack is a prerequisite, not tested by the benchmark smoke run.

## Outputs and interpretation

Raw arrays, optimizer/RNG states, cached adapters and full traces remain in ignored `runs/RUN_ID`. Small exports appear in `results/RUN_ID`: CSV/JSON metrics, 95% Wilson intervals, actual-count ledger, success-vs-rollout figure (PNG/PDF), error/cost/anchor plots, source/feedback provenance and representative videos. CUDA-event elapsed is a timeline proxy, not exact exclusive GPU-active compute; Torch peak memory excludes some EGL/driver allocation. Shared preparation, real cache savings and equivalent uncached training costs are separate. Concurrent unrelated GPU work can affect timings.

Wilson intervals measure conditional evaluation uncertainty for one trained checkpoint. They do not measure training-seed variation or prove causality. Data selection, label changes and training can change together; inspect ablations rather than attributing a full-method result to one stage. Scores of zero, worse feedback results, missing methods/budgets and exhausted feasibility must be reported unchanged.

See [execution record](EXPERIMENTS.md) for completed versus still-running experiments and the exact published result paths.
