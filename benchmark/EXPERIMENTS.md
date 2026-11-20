# Execution record

## Completed execution validation

`benchmark_smoke_20261119T191355Z_1340243` completed all ten common-interface implementations at allowed K=0/1, with two distinct evaluation states and a 20-step horizon. There were **8 counted adaptation episodes, 0 development episodes and 24 actual evaluation episodes**, with no started-but-unfinished episodes. Identical checkpoints share trials through explicit cache aliases. All recorded task success rates were zero; the short horizon/sample count make these execution checks, not performance evidence.

[Measured smoke export](../results/benchmark_smoke_20261119T191355Z_1340243/README.md)

[Descriptive failure indicators and stage ablations](../results/benchmark_smoke_20261119T191355Z_1340243/ANALYSIS.md) are exported separately. All smoke success differences are zero; this does not show that the methods are equivalent. The real-horizon pilot does not use these final-evaluation observations for adaptation or tuning.

Validation: 71 tests passed with `CROSSROBOT_TEST_RUN=fast_20261105T182127Z .venv/bin/python -m pytest -q --ignore=tests/test_humanoid_fixed_arm.py`. This includes 13 new benchmark tests. The two earlier untracked humanoid physics tests were excluded; the new static FK test checks native geometry, fixed-body transforms, arm joint count and joint limits without generating uncounted task rollouts.

Two earlier setup/debug attempts were preserved locally. The first stopped on a generated-asset path issue before target physics. The second exercised method adaptation but the protocol hash guard refused final evaluation after implementation changes. Its physical counts remain in its own ledger; its weights/data are not reused in the formal pilot. These debug attempts are not presented as an independent benchmark result.

## Independent 50-start pilot

`benchmark_pilot_20261120T041649Z_1451454` was launched in detached tmux with the pilot profile: pretrained, matched fixed-source training and full feedback; K=0/1; one training seed; 50 distinct native evaluation starts; 220-step horizon and20-step stable native success. This is a limited subset of the requested matrix.

At the time this record was created, this run was in progress. Only completed measured evaluation rows may be published. The full K=2/4/8/16 comparison, 50-start inspired-baseline comparison and multi-training-seed repetition are not yet completed. No research superiority claim is made.

[Launch/accounting snapshot](PILOT_LAUNCH.json) records the time and actual episode starts observed while preparing this commit. It is not a completed performance report. The run continues under its detached supervisor; its measured export will be written to `results/benchmark_pilot_20261120T041649Z_1451454/` on completion or timeout. Concurrent unrelated GPU work was present at launch, so timings are contended rather than isolated GPU measurements.

The timestamp in the run ID is UTC (the launch occurred on November19 in America/New_York).
