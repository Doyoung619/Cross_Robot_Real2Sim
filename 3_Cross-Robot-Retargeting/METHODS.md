# Stage 3 methods and limits

The extractor preserves source EEF position/orientation, object state, actions, and timing information. It finds a closest-approach grasp candidate during positive source gripper commands, measures the EEF-minus-object translation, and estimates maximum object lift. The active transfer uses these grasp/lift summaries, not the complete source trajectory or source orientation dynamics.

The current object-relative displacement uses simulator-world-aligned axes for a cylindrical can. Target grip-site calibration clips source XY grasp offsets to ±2.5 cm and Z to −1.5 to +4 cm; lift clearance is bounded. These are explicit engineering assumptions. Eight hand-authored phases perform approach, grasp, close, lift, transport, lower, open, and retreat using live target feedback. Phase dwell and controller rules are not learned source behavior.

Both robots use the installed robosuite `BASIC/OSC_POSE` controller. The shared adapter maps normalized 4D task-space commands into the actual 7D action layout through named `create_action_vector` fields: XYZ delta, zero orientation delta, and gripper. One unit is 0.05 m per control step on the tested configuration. Target joint angles are never copied from Panda.

Origins remain distinct: `source_scripted` is the live Panda smoke branch; `source_retargeted` uses an identified source grasp/lift summary; `target_scripted_bootstrap` is a separately labeled fallback if transfer fails. Placement is checked from actual simulator state. A source HDF5 state replay is a separate visualization: it shows recorded source states and is not the live scripted Panda rollout.

## Current limitations

This is source-grounded waypoint retargeting with hand-coded phases, calibrated grip offsets, and privileged state. It is not full trajectory retargeting, learned source-policy transfer, or an ablation showing source geometry is necessary. The source orientation trace is stored, while target control holds the initial downward grip orientation.

## Next measurable experiment

Ablate the measured grasp/lift summaries against the target-only controller, test source-demo sensitivity, and evaluate richer object-frame/orientation trajectory transfer.

Detailed integrated discussion: [English technical report](../reports/english_technical_report_20261106/Cross_Robot_Real2Sim_Technical_Report.pdf). Visual evidence: [English case studies](../results/english_case_studies_20261106/README.md).
