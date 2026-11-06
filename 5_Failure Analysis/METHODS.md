# Stage 5 methods and limits

The deterministic analyzer reads saved object/EEF/gripper/contact/phase histories and final task success. It distinguishes no lift, loss of grasp/drop, release/placement issues, and horizon timeouts where supported. Ambiguous physical causes remain unknown. Diagnoses explicitly mark `diagnostic_oracle_used=true` because simulator state is privileged.

The historical `unreachable_waypoint` code means an early waypoint was not reached within the episode horizon. It is an operational symptom, not proof that the target pose is outside the robot's kinematic workspace. The report must be read together with phase, joint-limit, contact, and state evidence.

The feedback branch collects a small bounded set of additional successful target demonstrations on fresh training seeds, retrains, and evaluates the same held-out seeds as the baseline. It records both success counts and final XY placement error. No feasible teacher or no baseline yields an explicit intervention-not-executed reason. Fresh passes archive old correction outputs so stale feedback trajectories cannot join a new dataset.

The case-study videos are illustrative frozen-model episode replays. Post-release stability checks extend visual success evidence with additional simulated hold time, but a few stable videos are not a full reliability study. New evaluation seed sets must be reported separately from the seeds used to choose a correction.

## Current limitations

The corrective iteration is a bounded heuristic experiment. Fast success stayed 9/10 and mean XY error rose from about 0.0815 m to 0.0846 m; there is no measured fast improvement. Small bootstrap gains do not establish statistical significance. No learned causal diagnosis or human feedback loop is implemented.

## Next measurable experiment

Use independent seed sets for intervention selection and final evaluation; add source-summary ablations, failure-focused collection, and longer post-release stability checks.

Detailed integrated discussion: [English technical report](../reports/english_technical_report_20261106/Cross_Robot_Real2Sim_Technical_Report.pdf). Visual evidence: [English case studies](../results/english_case_studies_20261106/README.md).
