# Stage 2 methods and limits

Bootstrap and fast modes load `data/task_seeds/can_to_bin.json` and validate it with the shared Pydantic `TaskSpec`. The task is Can → inside → bin, with named manipulation phases and constraints. The contract marks the source as `provided_seed`, and metric target placement as `simulator_oracle`.

Full mode uses `Qwen/Qwen2.5-VL-3B-Instruct` on a small ordered frame set. The implementation preserves raw output, parses structured JSON, validates supported task semantics, and explicitly maps semantic `can` to simulator `Can`, and `bin` to the can's target bin placement. It does not convert monocular image motion into a measured metric hand trajectory.

Unknown mass, friction, and visual contact remain `unknown`. Actual simulator grasp contact is privileged downstream state, distinct from video evidence. A missing full-mode clip yields `requires_human_video` and no task interpretation success. Unsupported goals are rejected rather than silently rewritten as can-inside-bin.

## Current limitations

Real CLIP/Qwen processing has not been validated in the recorded no-human-video experiments. Optional SAM2 segmentation is not implemented. A semantic task JSON alone does not recover metric scene geometry or physical parameters.

## Next measurable experiment

Supply a real human can-to-bin clip, inspect the raw/parsed Qwen goal against manual labels, and test whether parsing errors propagate into retargeting.

Detailed integrated discussion: [English technical report](../reports/english_technical_report_20261106/Cross_Robot_Real2Sim_Technical_Report.pdf). Visual evidence: [English case studies](../results/english_case_studies_20261106/README.md).
