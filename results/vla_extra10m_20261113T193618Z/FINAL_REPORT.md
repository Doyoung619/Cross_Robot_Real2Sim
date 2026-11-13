# Ten-minute B adapter continuation

Task: push the plate to the front of the stove. The prior 100-update adapter weights are the starting point.

Extra training: **600.06 seconds**, **8130 new updates**, **8230 total updates**.

The same 189 source-derived static-rendering anchors from three A episodes are reused. No executed B training demonstrations or new data are added. LoRA targets, learning rate, batch size, cameras, normalizers and inference settings remain fixed.

| B policy | Successes / trials | Mean final XY goal error |
|---|---:|---:|
| Previous 100 updates (re-evaluated) | 0/2 | 10.95 cm |
| Plus ten minutes | 0/2 | 20.00 cm |

## English videos

- [demo_45: previous vs extended training](videos/training_duration_comparison_demo_45.mp4)
- [demo_46: previous vs extended training](videos/training_duration_comparison_demo_46.mp4)

## Interpretation limits

The initial pilot saved adapter weights but no optimizer state. This continuation therefore starts from those exact weights with a fresh AdamW optimizer; it is not an exact optimizer/RNG resume. New checkpoints include optimizer and RNG state. Uniform sampling is with replacement, so update counts and equivalent sample exposures are reported instead of exact epochs.

The checkpoint is the last one at the fixed time budget, never the best target-score checkpoint. Target results are evaluation only. These same two test scenes were already examined; the result cannot establish broad generalization or isolate optimizer-state effects. More updates also cannot validate contact dynamics missing from the synthetic source processing.

Parent adapter SHA256: `69bbd848539c89c2b534d846fbe8fecc99683fb22c2e810ba6d903f2c640f428`.

New adapter SHA256: `0c0bbb4d7aab26e0569305e4e8b581eb808902a24577a2196154c2713d906409`.

Processed training data SHA256: `3732d96f46eafb9ac4e82c4e1d0754dfc56a03dae43d59b6ac7f767eb4daabe9`.


## Additional report artifacts

[English PDF](TEN_MINUTE_TRAINING_REPORT.pdf) · [Training loss](training_loss.png) · [Verification](verification.json).

Reproduce with `bash scripts/launch_vla_extra_training.sh` after the original pilot materials are installed. The tmux job performs 600 seconds of additional training, fixed evaluation and report export independently of SSH.
