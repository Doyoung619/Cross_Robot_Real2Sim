# Executed experiments

New review package: [watching guide](review_20261105/README.md), [fast five-stage video](review_20261105/videos/01_pipeline_fast.mp4), [bootstrap feedback failure-to-success video](review_20261105/videos/02_pipeline_bootstrap_feedback.mp4), [all nine fast-policy successes](review_20261105/videos/03_fast_success_gallery.mp4), and [current issue specification](review_20261105/CURRENT_ISSUES.md). The new videos label the no-human-video path, use frozen controllers/checkpoints, and add a one-second stability check with the same policy. They do not represent a completed human-video full run.

All results below come from actual headless MuJoCo rollouts on the tested RTX 4090 host. Inputs use **oracle_sim_state**, a provided task seed, and public robomimic Panda demonstrations. No real human video was supplied. Teacher, scripted-source, and learned-policy success are separate measurements.

| Run | Panda scripted | Genuine UR5e retargeting | BC before → after one correction | Mean XY error before → after |
|---|---:|---:|---:|---:|
| [bootstrap_20261105T181834Z](bootstrap_20261105T181834Z/FINAL_REPORT.md) | 1/1 | 3/3 | 0/3 → 1/3 | 0.653819 → 0.258879 m |
| [fast_20261105T182127Z](fast_20261105T182127Z/FINAL_REPORT.md) | 1/1 | 10/10 | 9/10 → 9/10 | 0.081546 → 0.084578 m |
| [full_20261105T182257Z](full_20261105T182257Z/FINAL_REPORT.md) | not run | not run | not run | `requires_human_video` |

Bootstrap trained on 3 executed source-retargeted UR5e teachers for 300 steps, then added 2 successful teachers and retrained for 300 steps. Fast trained on 10 teachers for 2,000 steps, then added 3 successful teachers and retrained for 2,000 steps. Each correction used the **identical held-out seeds** as its baseline. No independent target-scripted fallback was needed in these two runs. The fast correction did **not** improve success and slightly worsened mean error; no improvement is claimed. These small samples do not establish broad generalization.

Fast evidence:

- [Comparison video](fast_20261105T182127Z/videos/comparison.mp4): missing-human notice, scripted Panda, source-retargeted UR5e teacher, initial learned policy.
- [Learned policy after correction](fast_20261105T182127Z/05_diagnosis/feedback_run/videos/target_bc.mp4).
- [Teacher attempts](fast_20261105T182127Z/03_retarget/attempts.csv), [initial evaluation](fast_20261105T182127Z/04_policy/eval.csv), [corrected evaluation](fast_20261105T182127Z/05_diagnosis/feedback_run/04_policy/eval.csv).
- [Failure evidence](fast_20261105T182127Z/05_diagnosis/failure_report.csv), [feedback comparison](fast_20261105T182127Z/05_diagnosis/before_after.json), [export hashes](fast_20261105T182127Z/evidence_manifest.json).

Verification executed: **33 pytest tests passed**, actual Panda/UR5e three-step EGL preflight passed, isolated tmux failure/duplicate-launch/stop checks passed, and a subsequent fast resume verified and skipped all five complete stages. The full no-video branch ran and recorded `partial` / `requires_human_video`; CLIP/Qwen model inference remains untested. Optional SAM2 is not implemented.

Raw trajectories, checkpoints, history, logs, and source HDF5 remain in the local repository's gitignored `runs/` and `data/source/`. The compact exports contain measured CSV/JSON metadata, reports, and videos; `evidence_manifest.json` identifies their hashes. Regenerate an export with:

```bash
.venv/bin/python scripts/export_results.py RUN_ID
```
