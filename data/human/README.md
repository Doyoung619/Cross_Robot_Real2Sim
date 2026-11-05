Place an actual human can-pick-and-place `.mp4` clip here, then run:

```bash
bash scripts/setup.sh full
bash scripts/launch.sh full
```

Use short clips showing a person picking up a can and placing it inside a bin, with the object and destination visible. Do not put robot simulation renders here. Clips remain local and are gitignored. Full mode samples 4–6 ordered frames, runs CLIP ranking followed sequentially by Qwen2.5-VL-3B interpretation, saves raw model output, and validates the semantic goal. Full mode without a clip reports `requires_human_video`; bootstrap mode uses the explicitly provided task seed and simulator object poses.
