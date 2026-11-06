# Stage 1 methods and limits

The metadata baseline reads real HDF5 episode lengths and observation shapes. It keeps internally consistent Panda/Can episodes, starts from the shortest coherent candidate, and greedily adds the candidate whose initial object position is farthest from the already selected set. Candidate limits come from YAML; this is a reproducible small-data baseline, not an optimal task-aware retrieval method.

The optional full path samples ordered frames from actual human MP4s and computes image/text similarity with `openai/clip-vit-base-patch32`. It records numerical scores only after inference. HDF5 robot observations are not relabeled as human video. CLIP is released before the next GPU model.

The 200-episode source file is `robomimic/robomimic_datasets/v1.5/can/ph/low_dim_v15.hdf5` (46,889,752 bytes). The parser checks actual arrays, including EEF positions/quaternions, gripper/joint observations, and the 14D object observation. The object's world XYZ occupies indices `7:10`, and its quaternion occupies `10:14`; the preceding values are EEF-relative pose.

## Current limitations

No human input means `human_clip_selection=skipped_missing_input`; it does not count as a successful CLIP experiment. Start-position diversity is a heuristic and does not establish visual or semantic demonstration quality.

## Next measurable experiment

Compare metadata-only selection with actual CLIP rankings on a small labeled set of human task and distractor clips.

Detailed integrated discussion: [English technical report](../reports/english_technical_report_20261106/Cross_Robot_Real2Sim_Technical_Report.pdf). Visual evidence: [English case studies](../results/english_case_studies_20261106/README.md).
