# Cross-Robot Real2Sim — bootstrap_20261105T181834Z

Execution: **completed_degraded**, exit `0`; mode `bootstrap`. State input: **oracle_sim_state**. Privileged simulator object poses/contact are used.

## Actual stage outcomes

| Stage | Outcome | Seconds | Evidence |
|---|---|---:|---|
| 1: stage1_selection | completed_degraded | 0.06 | Source metadata selection executed; absent/unrequested human perception is explicitly skipped |
| 2: stage2_task | completed_degraded | 0.00 | Provided task seed validated; human-video interpretation not run |
| 3: stage3_retarget | completed | 5.44 | 2 real source traces, 3 successful target demos |
| 4: stage4_policy | completed | 8.34 | Learned BC 0/3 held-out placements; fresh process |
| 5: stage5_diagnosis | completed | 10.59 | completed |

## Measured placements

| Controller/data origin | Successes / attempts |
|---|---:|
| Panda scripted source policy | 1 / 1 |
| UR5e genuine source-grounded waypoint retargeting | 3 / 3 |
| UR5e independent scripted fallback | 0 / 0 |
| UR5e learned BC, before correction | 0 / 3 |
| UR5e learned BC, after correction | 1 / 3 |

BC training samples: 496; feature dimension: 26; training steps: 300.
Held-out seeds: `[10000, 10001, 10002]`. Mean final XY placement error: `0.6538189241556394` m before, `0.25887867930478853` m after.
Correction: `completed`, additional teachers 2/2; same held-out seeds: `[10000, 10001, 10002]`. These small samples do not establish statistical generalization.

## Reproduce

```bash
bash scripts/setup.sh
bash scripts/preflight.sh
bash scripts/launch.sh bootstrap
bash scripts/status.sh bootstrap_20261105T181834Z
bash scripts/resume.sh bootstrap bootstrap_20261105T181834Z
tail -f runs/bootstrap_20261105T181834Z/logs/pipeline.log
bash scripts/stop.sh bootstrap_20261105T181834Z
```

## Environment and inputs

Python: `3.10.12 (main, Aug 31 2026, 10:18:17) [GCC 11.4.0]`. GPU: `NVIDIA GeForce RTX 4090, 550.163.01, 15241 MiB`.
Packages: `{"h5py": "3.14.0", "huggingface-hub": "0.36.0", "mujoco": "3.3.7", "numpy": "1.26.4", "pydantic": "2.12.3", "robosuite": "1.5.2", "torch": "2.6.0+cu124", "transformers": "not_installed"}`. Exact lock: `requirements-working-bootstrap.lock`.
Dataset: `/home/dheo/Cross_Robot_Real2Sim/data/source/v1.5/can/ph/low_dim_v15.hdf5`. Selected source IDs: `['demo_2', 'demo_0']`.

- Source: public robomimic Can / PH / v1.5 low_dim_v15.hdf5; source data is human robot teleoperation, not a human video.
- Bootstrap task source: explicitly provided manual task seed. Robot trajectories and placement are executed in MuJoCo.
- Real human video, CLIP, Qwen inference and optional SAM2 are untested unless a full-mode report explicitly records their execution.
- SAM2 is an unimplemented optional extension. No vision-only transfer, physical robot execution, or real-world transfer is claimed.
- Source EEF orientation is preserved in source traces; target OSC holds the initial downward grip orientation, with disclosed grip-site calibration.
- BC uses a shared online event/geometry phase tracker; it is not an end-to-end vision policy. Timed gripper dwell is part of this controller.

## Evidence files

- [01_selection/selection.json](01_selection/selection.json)
- [02_task/task_spec.json](02_task/task_spec.json)
- `03_retarget/source_trace.npz` (kept in the raw local run)
- [03_retarget/attempts.csv](03_retarget/attempts.csv)
- [04_policy/dataset_metadata.json](04_policy/dataset_metadata.json)
- [04_policy/eval.csv](04_policy/eval.csv)
- [05_diagnosis/failure_report.csv](05_diagnosis/failure_report.csv)
- [05_diagnosis/feedback_requests.json](05_diagnosis/feedback_requests.json)
- [05_diagnosis/before_after.json](05_diagnosis/before_after.json)
- [comparison.mp4](videos/comparison.mp4) (264157 bytes)
- [source.mp4](videos/source.mp4) (87724 bytes)
- [target_bc.mp4](videos/target_bc.mp4) (88726 bytes)
- [target_teacher.mp4](videos/target_teacher.mp4) (90356 bytes)

## Ranked bottlenecks

- unreachable_waypoint: 3 observed episode(s). Check workspace limits and source object-relative offsets; use reachable target starts.
- Limited training coverage: 3 successful target teachers from 2 selected source demos. Hypothesis: more executed state coverage reduces BC distribution shift; test on new seeds.
- Human-video branch: `skipped_missing_input`. A real can-to-bin human clip is needed to measure CLIP/Qwen selection and goal-parsing errors.
