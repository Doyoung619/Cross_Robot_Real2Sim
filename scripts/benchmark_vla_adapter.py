#!/usr/bin/env python3
"""Time real pretrained SmolVLA LoRA updates on source (Panda) data.

This is a throughput smoke test, NOT target-robot training or a success eval.
"""
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/vla_readiness_20261113"
sys.path.insert(0, str(OUT / "deps"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    import argparse
    import h5py
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from peft import LoraConfig, inject_adapter_in_model
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1, choices=(1, 2, 4, 8))
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(42)
    started = time.monotonic()
    config = PreTrainedConfig.from_pretrained(OUT / "model")
    config.vlm_model_name = str(OUT / "backbone")
    config.load_vlm_weights = False
    config.device = "cpu"
    config.train_expert_only = True
    config.freeze_vision_encoder = True
    config.train_state_proj = False
    print("Constructing model from local config", flush=True)
    policy = SmolVLAPolicy(config)
    state = load_file(str(OUT / "model/model.safetensors"))
    matched = policy.load_state_dict(state, strict=True)
    del state
    policy.requires_grad_(False)
    targets = r".*(?:lm_expert\..*\.(?:q_proj|v_proj)|action_in_proj|action_out_proj)"
    adapter_config = LoraConfig(r=8, lora_alpha=16, target_modules=targets,
                                lora_dropout=0.0, bias="none")
    policy = inject_adapter_in_model(adapter_config, policy)
    policy = policy.to(device="cuda", dtype=torch.bfloat16)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    trainable = sum(p.numel() for p in parameters)
    print("Loaded strict checkpoint; trainable parameters", trainable, flush=True)
    hdf = next((OUT / "dataset/libero_spatial").glob("*.hdf5"))
    with h5py.File(hdf) as f:
        demo = f["data/demo_0"]
        obs = demo["obs"]
        state = np.r_[obs["ee_states"][0], obs["gripper_states"][0]].astype(np.float32)
        actions = np.asarray(demo["actions"][:50], dtype=np.float32)
        frames = [np.asarray(obs[name][0])[::-1].copy()
                  for name in ("agentview_rgb", "eye_in_hand_rgb")]
    stats = json.loads((OUT / "dataset/meta/stats.json").read_text())
    def normalize(x, key):
        d = stats[key]
        return (x - np.array(d["mean"])) / np.maximum(np.array(d["std"]), 1e-6)
    batch = {
        "observation.state": torch.tensor(normalize(state, "observation.state")[None], device="cuda", dtype=torch.bfloat16),
        "action": torch.tensor(normalize(actions, "action")[None], device="cuda", dtype=torch.bfloat16),
    }
    for i, frame in enumerate(frames, 1):
        batch[f"observation.images.camera{i}"] = torch.tensor(frame.transpose(2, 0, 1)[None] / 255., device="cuda", dtype=torch.bfloat16)
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    text = "pick up the black bowl from table center and place it on the plate\n"
    tokens = tokenizer([text], return_tensors="pt", padding="max_length", max_length=48, truncation=True)
    batch["observation.language.tokens"] = tokens["input_ids"].cuda()
    batch["observation.language.attention_mask"] = tokens["attention_mask"].bool().cuda()
    if args.batch_size > 1:
        batch = {key: value.repeat(args.batch_size, *([1] * (value.ndim-1)))
                 for key, value in batch.items()}
    policy.train()
    durations, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(8):
        torch.cuda.synchronize()
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite benchmark loss")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
        losses.append(float(loss.detach()))
        if step >= 2:
            durations.append(elapsed)
        print(json.dumps(dict(step=step+1, seconds=elapsed, loss=losses[-1])), flush=True)
    result = dict(status="completed", scope="source_Panda_throughput_only_not_B_training",
                  checkpoint_strict_load=True, batch_size=args.batch_size, action_chunk=50,
                  resize=list(config.resize_imgs_with_padding), warmup_steps=2, measured_steps=6,
                  seconds_per_step=float(np.median(durations)), seconds_each=durations,
                  total_parameters=sum(p.numel() for p in policy.parameters()),
                  trainable_parameters=trainable, lora_rank=8,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  losses=losses, duration_seconds=time.monotonic()-started,
                  caveats=["Repeated single source batch; excludes data loading and checkpoint I/O",
                           "Original HDF5 128px images; production preprocessing still requires validation",
                           "Other GPU workload active; not a source or target policy success evaluation"])
    result_path = OUT / ("benchmark.json" if args.batch_size == 1 else f"benchmark_batch{args.batch_size}.json")
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
