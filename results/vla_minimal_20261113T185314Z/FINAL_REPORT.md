# One-hour source-only VLA transfer pilot

Task: **push the plate to the front of the stove**.

A: Panda (7 arm joints). B: UR5e (6 arm joints), Robotiq85 gripper.

Training uses only existing source records transformed by offline B kinematics and static rendering. **No executed B training demonstrations.**

This is a minimal feasibility experiment, not evidence of broad generalization.

## Measured final evaluation

| Condition | Successful / evaluated | Mean final XY goal error |
|---|---:|---:|
| A_pretrained_Panda | 1/2 | 0.0813 m |
| B_pretrained_UR5e | 0/2 | 0.2047 m |
| B_adapted_UR5e | 0/2 | 0.1095 m |

A source-policy success is a prerequisite to claiming transfer. Positive human-demo terminal rewards do not prove VLA source competence.

Measured adapted-minus-unadapted B success-rate difference: **+0.000**. With only 2 trials, this is descriptive, not a reliable performance estimate.

## Data and model provenance

Source selection: `{'candidate_count': 45, 'eligible_count': 35, 'selected': ['demo_5', 'demo_38', 'demo_26'], 'evaluation_source_ids': ['demo_45', 'demo_46'], 'excluded_validation_ids': ['demo_45', 'demo_46', 'demo_47', 'demo_48', 'demo_49'], 'target_rollout_training_calls': 0, 'method': 'four_pose_B_IK_and_static_arm_collision_filter_then_shortest_and_farthest_plate_starts'}`.

Adapter training: `{'status': 'completed', 'steps': 100, 'training_frames': 189, 'batch_size': 4, 'trainable_parameters': 334592, 'rank': 8, 'learning_rate': 0.0001, 'source_checkpoint': 'lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d', 'normalization': 'exact_saved_source_checkpoint_normalizer', 'training_origin': 'source_derived_static_kinematic_render_only', 'executed_target_training_episodes': 0, 'duration_seconds': 17.563157806172967, 'adapter_sha256': '69bbd848539c89c2b534d846fbe8fecc99683fb22c2e810ba6d903f2c640f428', 'peak_allocated_gib': 1.5862503051757812, 'final_training_loss': 0.09566406160593033, 'source_only_validation': 'not_performed_in_minimal_compute_smoke'}`.

The frozen pretrained checkpoint is lerobot/smolvla_libero revision a24f1f380461b665158d52acd140152c79e8b36d. Source data is original LIBERO plate-pushing demonstrations.

Static rendering copies source object states and solves hypothetical B poses. It does not establish physically correct B contact dynamics. Synthetic images are labeled as source-derived, not B execution recordings.

Both B conditions use the same fixed source-frame camera/state calibration. The pretrained-B baseline has no learned weight adaptation. Evaluation outputs are never read by training.

Control: world-frame relative OSC_POSE, 20 Hz, 7D Cartesian+gripper action. Three source episodes maximum, 100 LoRA updates, batch four, rank eight. Source state records provide the geometry; synthetic training and live evaluation render 256px images with the official 180-degree image rotation, followed by the saved 512px VLA preprocessing. No human video reconstruction.

No independent hyperparameter search or source-validation checkpoint selection in this minimal run. An unselected-source control is also absent, so selection benefit cannot be isolated.

## English comparison videos

- [Comparison: demo_45](videos/comparison_demo_45.mp4)
- [Comparison: demo_46](videos/comparison_demo_46.mp4)
- [Offline source processing](videos/offline_source_processing.mp4)
- [English PDF report](SOURCE_ONLY_VLA_PILOT_REPORT.pdf)

## Individual final rollout videos

- [A_pretrained_Panda / demo_45 / success=True](videos/A_pretrained_Panda_demo_45.mp4)
- [A_pretrained_Panda / demo_46 / success=False](videos/A_pretrained_Panda_demo_46.mp4)
- [B_pretrained_UR5e / demo_45 / success=False](videos/B_pretrained_UR5e_demo_45.mp4)
- [B_pretrained_UR5e / demo_46 / success=False](videos/B_pretrained_UR5e_demo_46.mp4)
- [B_adapted_UR5e / demo_45 / success=False](videos/B_adapted_UR5e_demo_45.mp4)
- [B_adapted_UR5e / demo_46 / success=False](videos/B_adapted_UR5e_demo_46.mp4)

## Audit artifacts

- `selection_candidates.json`, `selection.json`: offline B-aware selection.
- `processing_audit.json`: per-frame IK checks and source lineage; zero target physics steps.
- `training.json`, `train_log.json`, `adapter.safetensors`: source-initialized LoRA update evidence.
- `evaluation.json`, `*_actions.json`: final closed-loop results, never training inputs.
- `worker.log`, `progress.json`, `status.json`: execution and time-budget evidence.


## Reproduce and survive SSH disconnect

```bash
bash scripts/launch_source_only_vla_pilot.sh
```

The detached tmux supervisor enforces a maximum of sixty minutes and records a report on failure or timeout. This run completed; no background training remains. The source materials and isolated runtime must already be installed.

The first image-contract run is archived locally and excluded. The corrected run retains all six final outcomes, including failures.

The source VLA achieved only 1/2 successes, so this pilot does not establish reliable transfer from an already competent source. Synthetic future action chunks retain A Cartesian commands; only current training anchors receive residual TCP correction and static feasibility filtering.
