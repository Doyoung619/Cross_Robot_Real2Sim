# B VLA failure-feedback experiment

Task: push the plate to the front of the stove. Starting point: the preserved 100-update B LoRA adapter, not the ten-minute continuation.

B failure rollouts now provide diagnostic feedback for choosing existing A source data and changing offline retargeting/action labels. This is a different protocol from the earlier source-only, evaluation-only experiment.

The checkpoint selected on development scenes was **round1 (200 total updates)**. Reserved final success changed **0/2 to1/2**: scene47 succeeds (3.77cm residual), while scene48 still fails (24.99cm). These two cases do not establish robust transfer.

## Measured round-by-round outcomes

| Stage | Role | Success / episodes | Mean final XY error | Mean plate-contact fraction |
|---|---|---:|---:|---:|
| baseline | Development feedback | 0/2 | 26.44 cm | 9.09% |
| round_1 | Development feedback | 0/2 | 23.10 cm | 5.00% |
| round_2 | Development feedback | 0/2 | 26.24 cm | 2.95% |
| round_3 | Development feedback | 0/2 | 26.75 cm | 10.68% |
| final:final_original100 | Reserved final evaluation | 0/2 | 24.54 cm | 37.27% |
| final:final_selected_round_1 | Reserved final evaluation | 1/2 | 14.38 cm | 30.64% |
| control:no_feedback_control | Compute-matched control | 0/2 | 25.08 cm | 1.14% |

Development scenes are intentionally reused to guide the feedback loop. Their improvements are development measurements, not independent generalization evidence. Reserved final scenes are listed separately and should not guide interventions.

## Source data and update budget

The original training set contains189 anchors from A demos5,38,26. Each round adds two previously unused A episodes.

| Round | New A demos | New anchors | Total anchors | Total optimizer updates |
|---|---|---:|---:|---:|
| 1 | 37,36 | 138 | 327 | 200 |
| 2 | 14,33 | 123 | 450 | 300 |
| 3 | 10,28 | 142 | 592 | 400 |

Each round runs100 optimizer updates, rather than100 full-dataset epochs. The selected checkpoint therefore uses five A episodes and327 anchors. No B action trace is included as a supervised target.

## What each feedback round changed

### round_1

Measured contact_without_sufficient_plate_response feeds source retrieval, bounded static contact optimization and full action-chunk recalculation.

Selected source IDs: demo_5, demo_38, demo_26, demo_37, demo_36.

Recorded update count: 100. Exact training provenance is in [rounds/round_1/training.json](rounds/round_1/training.json).

[Measured rollout diagnosis](rounds/round_1/diagnosis.json). These are evidence-based failure hypotheses, not proven causal diagnoses.

### round_2

Measured early_progress_then_stall feeds source retrieval, bounded static contact optimization and full action-chunk recalculation.

Selected source IDs: demo_5, demo_38, demo_26, demo_37, demo_36, demo_14, demo_33.

Recorded update count: 100. Exact training provenance is in [rounds/round_2/training.json](rounds/round_2/training.json).

[Measured rollout diagnosis](rounds/round_2/diagnosis.json). These are evidence-based failure hypotheses, not proven causal diagnoses.

### round_3

Measured insufficient_plate_contact feeds source retrieval, bounded static contact optimization and full action-chunk recalculation.

Selected source IDs: demo_5, demo_38, demo_26, demo_37, demo_36, demo_14, demo_33, demo_10, demo_28.

Recorded update count: 100. Exact training provenance is in [rounds/round_3/training.json](rounds/round_3/training.json).

[Measured rollout diagnosis](rounds/round_3/diagnosis.json). These are evidence-based failure hypotheses, not proven causal diagnoses.

No executed B trajectory is used as action supervision in the recorded updates. B rollout diagnostics influence source selection and offline processing; the source-derived frames and their labels remain the supervised data.

## English comparison movies

- [feedback_rounds_demo_40](videos/feedback_rounds_demo_40.mp4)
- [feedback_rounds_demo_41](videos/feedback_rounds_demo_41.mp4)

## Individual live rollouts

- [baseline / demo_40 / success=False](rounds/baseline/videos/baseline_100_demo_40.mp4)
- [baseline / demo_41 / success=False](rounds/baseline/videos/baseline_100_demo_41.mp4)
- [round_1 / demo_40 / success=False](rounds/round_1/videos/feedback_round_1_demo_40.mp4)
- [round_1 / demo_41 / success=False](rounds/round_1/videos/feedback_round_1_demo_41.mp4)
- [round_2 / demo_40 / success=False](rounds/round_2/videos/feedback_round_2_demo_40.mp4)
- [round_2 / demo_41 / success=False](rounds/round_2/videos/feedback_round_2_demo_41.mp4)
- [round_3 / demo_40 / success=False](rounds/round_3/videos/feedback_round_3_demo_40.mp4)
- [round_3 / demo_41 / success=False](rounds/round_3/videos/feedback_round_3_demo_41.mp4)
- [final:final_original100 / demo_47 / success=False](final/videos/final_original100_demo_47.mp4)
- [final:final_original100 / demo_48 / success=False](final/videos/final_original100_demo_48.mp4)
- [final:final_selected_round_1 / demo_47 / success=True](final/videos/final_selected_round_1_demo_47.mp4)
- [final:final_selected_round_1 / demo_48 / success=False](final/videos/final_selected_round_1_demo_48.mp4)
- [control:no_feedback_control / demo_40 / success=False](control/videos/no_feedback_control_demo_40.mp4)
- [control:no_feedback_control / demo_41 / success=False](control/videos/no_feedback_control_demo_41.mp4)

## Interpretation limits

Each intervention changes data selection, offline labels and adapter weights together. A compute-matched no-feedback control is reported separately, which helps compare feedback-guided processing with additional updates on unchanged data. Selection and retargeting still change together, so their individual contributions are not isolated. A development improvement does not establish broad transfer, and a failed intervention does not disprove every proposed hypothesis.

Offline IK and static rendering do not validate target pushing friction or contact dynamics. The physically executed policy rollouts and native task success are the actual execution evidence. Trace, diagnosis, selection and processing files are preserved per round.

Retargeting uses a bounded static contact-offset search at three source poses and a smooth trajectory correction, not dynamics-aware trajectory optimization. New action labels use future TCP displacement over a response-conditioned 2–5-frame lookahead; this is heuristic controller inversion and also changes action magnitude. It is not a calibrated inverse of the OSC dynamics.

[English PDF](FEEDBACK_REPORT.pdf).

## Preserved correction record

A first execution attempt was corrected before the reported round comparison. Its documented correction is retained in [INPUT_CONTRACT_CORRECTION.md](INPUT_CONTRACT_CORRECTION.md); the archived attempt is not silently mixed into the current scores.


## Reserved final before/after comparison videos

- [Final demo_47: original 100 updates vs selected feedback round 1](videos/final_comparison_demo_47.mp4)
- [Final demo_48: original 100 updates vs selected feedback round 1](videos/final_comparison_demo_48.mp4)

These show both reserved scenes, with the same initial states. Successful rollouts end after twenty consecutive successful steps; their terminal image is held when the other panel continues. No failed case is omitted.
