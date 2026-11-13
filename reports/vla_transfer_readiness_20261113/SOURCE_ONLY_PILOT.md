# Fastest source-only VLA transfer pilot

This is an engineering proposal, not an executed transfer result. It narrows
the initial experiment to **push the plate to the front of the stove** in
LIBERO-Goal. Panda is A and UR5e is B. Pushing avoids the pickup/lift/release
sequence and is a reasonable first feasibility test; its source-policy success
and cross-robot transfer success remain unmeasured.

## Actual materials

The additional original task HDF5 has been downloaded and inspected:
50 source demonstrations, 7,638 steps, 50 positive terminal rewards,
762,855,139 bytes. [Pinned file manifest](push_pilot_material.json).
The existing pretrained checkpoint remains `lerobot/smolvla_libero`; its saved
training configuration identifies the 40-task `lerobot/libero` training dataset,
whose task metadata includes this instruction. This does not establish the
checkpoint's closed-loop task success. Keep the original bowl pilot's assets
as an alternative, but start with this simpler interaction.

## Protocol and processing

1. Integrate the matching LIBERO environment and verify A on ten fixed starts.
   Measure direct, unadapted deployment on B as a baseline. B test actions and
   outcomes are not demonstrations or adaptation targets.
2. Split source episodes before selection (40 candidate training / 10 source
   validation). Within the forty, rank by coherent task completion, B's offline
   EEF pose reachability, joint-limit margin, unintended collision clearance,
   compatible pushing contact geometry and start/motion diversity. Select ten
   feasible source episodes, or explicitly report insufficient eligible data.
   The B-aware ranking and split have not been implemented yet.
3. Extract the source's full EEF position/orientation, object motion, gripper
   state and timestamps. Convert through calibrated object/world/TCP frames.
   Use B's offline IK/FK with joint continuity and velocity checks. Preserve
   intended pushing contact while rejecting unintended arm/table collisions.
   If the sample rate changes, reconstruct actions for that interval and
   controller contract rather than simply dropping action rows.
4. A practical observation-transfer candidate is **static kinematic rendering**:
   at each source timestamp set object poses from the A recording and B joints
   from offline IK; render external/wrist images without advancing physics or
   running a B policy. Label these as source-derived synthetic observations.
   Pair with B's hypothetical proprioception and derived control targets, not
   observed successful B actions. This rendering is offline processing of A,
   not acquisition of B executed demonstrations. It is not implemented yet.
5. Load the original A VLA weights, freeze the visual/language backbone and
   fine-tune rank-eight LoRA on the action expert/projections for 2,000 updates
   initially, capped at 5,000. Validate on source-derived held-out episodes and
   kinematic constraints. Preserve original source IDs and transformation logs.
6. Freeze preprocessing, data selection and weights. Evaluate A, unadapted B and
   adapted B on ten fixed starts each, with native task success, goal distance,
   unintended collision rate, seed-level logs and English videos. Never use
   these final target trajectories or rewards as training labels or selectors.

Static rendering addresses appearance only. It does not prove feasible pushing
forces, friction, contact timing or target dynamic success. The first experiment
can fail; reporting that failure is still a completed feasibility experiment.
If A itself fails, reassess the source checkpoint/task before claiming transfer.

## Timing

| Remaining work | Engineering estimate |
|---|---:|
| Simulator integration and source-policy verification | 1–2 hours |
| B-aware selection and offline retargeting/rendering implementation | 2–4 hours |
| Dataset construction, training integration and checks | 30–60 minutes |
| Initial 2,000–5,000 adapter updates including loading/checkpoints | 10–45 minutes |
| Baseline/adapted evaluation and evidence packaging | 30–90 minutes |

**Plan for 6–12 hours to the first complete success/failure comparison.**
An optimistic 4–6 hours assumes smooth integration; unresolved visual/control
compatibility may extend this beyond one working day. These are unmeasured
engineering estimates, not guaranteed completion or improvement.

The only measured throughput is [batch-eight model-only timing](benchmark_batch8.json):
0.08915 seconds/update on repeated Panda data. This gives 3.0–7.4 minutes for
2,000–5,000 model updates, excluding all data/conversion/checkpoint overhead.
Re-measure with the actual processed dataset. Human-video reconstruction is
excluded from this first pilot because it adds unverified 3D/contact inference.

To establish that B-aware selection helps, a subsequent comparison must train
on a random ten-source-episode subset with the same processing, update budget
and evaluation protocol. The first before/after comparison alone cannot isolate
the contribution of data selection, and ten trials give only a coarse estimate.

References: [official task definition](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_goal/push_the_plate_to_the_front_of_the_stove.bddl),
[pretrained policy](https://huggingface.co/lerobot/smolvla_libero).
