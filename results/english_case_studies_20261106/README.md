# English case studies: original Robot A motion, B training data, learned B execution

These three videos show **actual recorded Panda source motion**, the **executed UR5e teacher data generated from that source**, and a **successful held-out rollout of the learned UR5e policy**. Every title, overlay, and caption in the new videos is English.

## Watch the three cases

| Case | Original Robot A source | B training demonstration shown | Learned B evaluation shown | Full English video |
|---|---|---|---|---|
| 01 | Panda `demo_3`, 98 recorded states / 4.90 s | UR5e seed `1042`, 158 state-action pairs | Shared BC policy, held-out seed `10001` | [Case 01 - 58.1 s](videos/case_01_demo_3_to_ur5e.mp4) |
| 02 | Panda `demo_5`, 134 recorded states / 6.70 s | UR5e seed `1043`, 162 state-action pairs | Shared BC policy, held-out seed `10003` | [Case 02 - 60.6 s](videos/case_02_demo_5_to_ur5e.mp4) |
| 03 | Panda `demo_7`, 115 recorded states / 5.75 s | UR5e seed `1044`, 175 state-action pairs | Shared BC policy, held-out seed `10009` | [Case 03 - 67.05 s](videos/case_03_demo_7_to_ur5e.mp4) |

Each movie first identifies the source and target episodes, shows the original A motion, displays the complete B training lineage, shows the actual B teacher and learned B separately, and finishes with an A/teacher/learned comparison and measured outcomes. The green target outline appears in the B renders. Ended panels are explicitly labeled when their last frame is held.

## What exactly is the original A video?

The source file is public robomimic **Can / PH / v1.5 / low-dimensional HDF5**, SHA256 `3f2eb92e0a5025d0095e866ac16cc8092d6a762abe27dec90dbaff9027282962`. The three source episodes each have a recorded terminal reward of 1.0 and terminal done flag of 1.

The dataset contains numerical states rather than source RGB videos. The new A videos reconstruct **every original simulator state using that episode's original MJCF XML** and original camera. Only public asset file paths are resolved to the installed robosuite package. No new scripted controller is substituted, and the old actions are not physically re-simulated. Source timing remains 20 Hz at 1x speed.

Every reconstructed frame was compared with the original observations: **347 frames total, Can world-position error 0 m, maximum EEF position error 6.89e-16 m**. Original source videos, hashes, and per-frame validation CSVs are under [source/](source/). This numerical validation establishes state-reconstruction fidelity; it is not a claim that original RGB camera footage was present in the low-dimensional dataset.

## Which demonstrations actually trained B?

**All three cases use ONE shared frozen BC checkpoint.** The displayed Panda demo is a source-lineage example, not the only source used for that case's model. The complete pool is:

| Original Panda demo | Executed UR5e teacher seeds | Training state-action samples |
|---|---|---:|
| `demo_3` | `1042`, `1047` | 158 + 170 = 328 |
| `demo_5` | `1043`, `1048` | 162 + 157 = 319 |
| `demo_7` | `1044`, `1049` | 175 + 156 = 331 |
| `demo_4` | `1045`, `1050` | 156 + 160 = 316 |
| `demo_1` | `1046`, `1051` | 151 + 175 = 326 |
| **Total** | **10 successful executed UR5e episodes** | **1,620** |

The Panda trajectories themselves are not fed directly into UR5e BC as target actions. Their grasp-offset and lift-clearance summaries guide a hand-coded target controller, which executes the ten target episodes. BC then learns from those **target** state-action pairs. The full source-to-teacher-to-model mapping, trajectory hashes, and checkpoint hash are in [training_lineage.csv](training_lineage.csv) and [case_manifest.json](case_manifest.json).

Shared checkpoint SHA256: `71148db0b7a100df0465191ce3329b0fa9a56faace25775495bcebbfde75220e`. Architecture: `26 -> 128 ReLU -> 128 ReLU -> 4`. Training: 2,000 Adam updates, batch size 128, training-only feature normalization. No new policy training was performed to produce these videos.

## What did B actually do?

| Case | Teacher terminal XY error | Learned B terminal XY error | Original/replay terminal position difference | Same-policy 1 s stable placement |
|---|---:|---:|---:|---|
| 01 | 7.40 mm | 8.48 mm | 0 m for both | Passed for both |
| 02 | 49.89 mm | 4.86 mm | 0 m for both | Passed for both |
| 03 | 11.87 mm | 10.33 mm | 0 m for both | Passed for both |

Task success means placement in the assigned target bin; the can need not be exactly at the bin center. Terminal XY error is distance to that center, so a 49.89 mm teacher error can still be an in-bin success. The original native success predicate and an additional one-second stability check are reported separately. During that extra second, the **same original teacher or learned policy** continues to act; no scripted action override helps the learned model finish.

The three learned examples were selected from the original fast policy's **9 successful evaluations out of 10**. They are not a new three-trial performance estimate. The remaining original evaluation, seed 10000, failed and remains documented in the technical report and [earlier failure comparison](../review_20261105/videos/04_failure_teacher_vs_policy.mp4).

## Scope and limitations

- Retargeting uses source-relative **grasp position and lift height**, grip-site clipping, hand-coded phases, and target OSC feedback. It does not transfer the source's complete joint, orientation, or timing trajectory.
- Source, teacher, and learned panels have independent initial states and time axes. Side-by-side playback does not assert temporal or joint-path correspondence.
- The learned model uses privileged simulator state and an online hand-coded phase tracker. At inference it receives no source video or source demonstration ID. Its tracker uses the first successful teacher's calibration parameters.
- No real human MP4 was supplied. CLIP/Qwen inference, human-video full transfer, source-free causal ablation, broader generalization, and physical robot transfer remain untested.
- Frozen original experiments and checkpoints remain unchanged. All new reconstructions and replays are review evidence and are excluded from training.

## Detailed methods and stage navigation

[Download the English technical report (PDF)](../../reports/english_technical_report_20261106/Cross_Robot_Real2Sim_Technical_Report.pdf). It details models, datasets, algorithms, contracts, measured outcomes, known failures, limitations, and proposed acceptance gates across all five stages.

1. [Data Selection](<../../1_Data Selection/README.md>)
2. [Task Understanding](<../../2_Task Understanding/README.md>)
3. [Cross-Robot Retargeting](../../3_Cross-Robot-Retargeting/README.md)
4. [Policy Learning](<../../4_Policy Learning/README.md>)
5. [Failure Analysis](<../../5_Failure Analysis/README.md>)

## Reproduce and inspect

```bash
cd /home/dheo/Cross_Robot_Real2Sim
# Requires the source HDF5 and frozen original run artifacts/checkpoint.
bash scripts/launch_english_cases.sh all
cat runs/english_cases_20261106/status.json
tail -f runs/english_cases_20261106/logs/english_cases.log

# Rebuild the embedded-font PDF and run its layout/text audit.
.venv/bin/python -m pip install -r requirements-report.txt
.venv/bin/python scripts/build_english_report.py
```

Authentic A reconstruction is provided by [render_source_demos.py](../../scripts/render_source_demos.py). Its detached run is recorded under `runs/english_source_replay_20261106/`. B rendering and case composition use [build_english_cases.py](../../scripts/build_english_cases.py), with raw replay NPZs under gitignored `runs/english_cases_20261106/rollouts/`.

Published artifacts include English MP4 files, captions/transcripts, source validation CSVs, state/model/video hashes, and per-case provenance. Original source HDF5, model checkpoints, and trajectory NPZs remain local and gitignored; a fresh clone must prepare those before regenerating the movies.
