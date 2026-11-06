# Stage 4 methods and limits

The dataset builder accepts only successful UR5e episodes labeled `source_retargeted` or `target_scripted_bootstrap`; origin labels remain in provenance. It rejects non-finite arrays, inconsistent feature sizes, out-of-range normalized actions, and any training/evaluation seed overlap. No random targets or failed demonstrations are substituted when successful teacher data are absent.

The shared 26D observable feature vector contains EEF/object/goal XYZ, relative object/goal offsets, gripper opening, grasp contact, object lift, and eight phase indicators. Collection and inference use the same online event/geometry tracker. That tracker and privileged grasp/object state are part of the controller; evaluation does not receive expert-only trajectory phase labels.

The policy is an MLP with two 128-unit ReLU layers and four outputs. Training uses XYZ MSE plus gripper BCE, with feature normalization fitted on training data only. Checkpoints include model/optimizer state, normalization, model configuration, seed, and training progress. Learned outputs pass through the same robot action adapter as teacher data.

Fast mode uses 2,000 optimization steps and evaluates seeds `10000`–`10009`. A separate Python process loads the saved checkpoint before actual simulator rollouts. One original fast checkpoint is shared by all corresponding case-study episodes; multiple videos do not imply multiple learned models.

## Current limitations

Fast performance is 9/10 on ten simulator seeds, with privileged state and hand-coded online phase machinery. This is small-sample evidence and does not establish visual generalization or real-world transfer. Three additional fast teachers preserved the 9/10 success count and slightly worsened mean placement error.

## Next measurable experiment

Evaluate new untouched seeds, compare phase-conditioned BC with alternative online state designs, and measure coverage versus additional teacher data.

Detailed integrated discussion: [English technical report](../reports/english_technical_report_20261106/Cross_Robot_Real2Sim_Technical_Report.pdf). Visual evidence: [English case studies](../results/english_case_studies_20261106/README.md).
