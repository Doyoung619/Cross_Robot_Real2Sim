# Cross-Robot Real2Sim — full_20261105T182257Z

Execution: **partial**, exit `0`; mode `full`. State input: **oracle_sim_state**. Privileged simulator object poses/contact are used.

## Actual stage outcomes

| Stage | Outcome | Seconds | Evidence |
|---|---|---:|---|
| 1: stage1_selection | completed_degraded | 0.07 | Source metadata selection executed; absent/unrequested human perception is explicitly skipped |
| 2: stage2_task | skipped_missing_input | 0.00 | requires_human_video |
| 3: stage3_retarget | skipped_missing_input | 0.00 | 0 real source traces, 0 successful target demos |
| 4: stage4_policy | skipped_missing_input | 0.00 | requires_human_video |
| 5: stage5_diagnosis | completed_degraded | 0.58 | intervention_not_executed |

## Measured placements

| Controller/data origin | Successes / attempts |
|---|---:|
| Panda scripted source policy | 0 / 0 |
| UR5e genuine source-grounded waypoint retargeting | 0 / 0 |
| UR5e independent scripted fallback | 0 / 0 |
| UR5e learned BC, before correction | not run / not run |
| UR5e learned BC, after correction | not run / not run |

Retargeting was not evaluated in this run: `requires_human_video`.

BC training samples: 0; feature dimension: not trained; training steps: 0.
Held-out seeds: `[]`. Mean final XY placement error: `None` m before, `None` m after.
Correction: `intervention_not_executed`, additional teachers 0/0; same held-out seeds: `[]`. These small samples do not establish statistical generalization.

## Reproduce

```bash
bash scripts/setup.sh
bash scripts/preflight.sh
bash scripts/launch.sh full
bash scripts/status.sh full_20261105T182257Z
bash scripts/resume.sh full full_20261105T182257Z
tail -f runs/full_20261105T182257Z/logs/pipeline.log
bash scripts/stop.sh full_20261105T182257Z
```

## Environment and inputs

Python: `3.10.12 (main, Aug 31 2026, 10:18:17) [GCC 11.4.0]`. GPU: `NVIDIA GeForce RTX 4090, 550.163.01, 15241 MiB`.
Packages: `{"h5py": "3.14.0", "huggingface-hub": "0.36.0", "mujoco": "3.3.7", "numpy": "1.26.4", "pydantic": "2.12.3", "robosuite": "1.5.2", "torch": "2.6.0+cu124", "transformers": "not_installed"}`. Exact lock: `requirements-working-bootstrap.lock`.
Dataset: `data/source/v1.5/can/ph/low_dim_v15.hdf5`. Selected source IDs: `[]`.

- Source: public robomimic Can / PH / v1.5 low_dim_v15.hdf5; source data is human robot teleoperation, not a human video.
- Bootstrap task source: explicitly provided manual task seed. Robot trajectories and placement are executed in MuJoCo.
- Real human video, CLIP, Qwen inference and optional SAM2 are untested unless a full-mode report explicitly records their execution.
- SAM2 is an unimplemented optional extension. No vision-only transfer, physical robot execution, or real-world transfer is claimed.
- Source EEF orientation is preserved in source traces; target OSC holds the initial downward grip orientation, with disclosed grip-site calibration.
- BC uses a shared online event/geometry phase tracker; it is not an end-to-end vision policy. Timed gripper dwell is part of this controller.

## Evidence files

- [01_selection/selection.json](01_selection/selection.json)
- [05_diagnosis/failure_report.csv](05_diagnosis/failure_report.csv)
- [05_diagnosis/feedback_requests.json](05_diagnosis/feedback_requests.json)
- [05_diagnosis/before_after.json](05_diagnosis/before_after.json)

## Ranked bottlenecks

- Limited training coverage: 0 successful target teachers from 0 selected source demos. Hypothesis: more executed state coverage reduces BC distribution shift; test on new seeds.
- Human-video branch: `requires_human_video`. A real can-to-bin human clip is needed to measure CLIP/Qwen selection and goal-parsing errors.
No video evidence was produced; see stage exception logs.
