"""Bounded BC optimization with checkpointed optimizer and training-only scaling."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .bc_model import BCPolicy


def train(config: dict, output_dir: Path, *, resume: bool = False) -> dict:
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    requested = str(config.get("device", "cuda:0"))
    device = requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu"
    with np.load(output_dir / "train_data.npz", allow_pickle=False) as data:
        x, y = data["features"], data["actions"]
        mean, std = data["mean"], data["std"]
        metadata = json.loads(str(data["metadata_json"].item()))
    model_config = {"feature_dim": int(x.shape[1]), "hidden_dim": int(config.get("bc_hidden_dim", 128))}
    model = BCPolicy(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("bc_learning_rate", 1e-3)))
    features = torch.as_tensor((x - mean) / std, dtype=torch.float32, device=device)
    targets = torch.as_tensor(y, dtype=torch.float32, device=device)
    close = (targets[:, 3] > 0).float()
    pos_weight = ((1 - close).sum() / close.sum().clamp_min(1)).clamp(0.25, 4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    steps = int(config.get("bc_train_steps", 300))
    if not 1 <= steps <= 10000:
        raise ValueError("bc_train_steps must be within the explicit compute budget [1,10000]")
    batch_size = min(int(config.get("bc_batch_size", 256)), len(x))
    start_step = 0
    checkpoint_path = output_dir / "checkpoint.pt"
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["dataset_metadata"] != metadata:
            raise ValueError("Cannot resume optimizer against changed training data")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if device.startswith("cuda") and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        start_step = int(saved["step"])
    generator = torch.Generator(device=device).manual_seed(seed)
    if resume and start_step and saved.get("sampling_rng_state") is not None:
        generator.set_state(saved["sampling_rng_state"].cpu())
    start_time = time.monotonic()
    last_loss = None

    def save(step):
        payload = {"schema_version": "1.0", "seed": seed, "mode": config.get("mode"),
                   "model_name": "MLP_BC_2x128", "model_config": model_config,
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "normalization": {"mean": mean.tolist(), "std": std.tolist()},
                   "dataset_metadata": metadata, "step": step, "device": device,
                   "torch_rng_state": torch.get_rng_state(), "sampling_rng_state": generator.get_state(),
                   "cuda_rng_state": torch.cuda.get_rng_state(device) if device.startswith("cuda") else None,
                   "privileged_state_used": True, "observation_mode": "oracle_sim_state"}
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)

    log_path = output_dir / "train_log.csv"
    append = resume and log_path.exists()
    with log_path.open("a" if append else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss", "xyz_mse", "gripper_bce", "seconds"])
        if not append:
            writer.writeheader()
        for step in range(start_step, steps):
            idx = torch.randint(len(x), (batch_size,), generator=generator, device=device)
            prediction = model(features[idx])
            xyz_loss = nn.functional.mse_loss(prediction[:, :3], targets[idx, :3])
            grip_loss = criterion(prediction[:, 3], close[idx])
            loss = xyz_loss + float(config.get("gripper_loss_weight", 0.1)) * grip_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            if step % 10 == 0 or step == steps - 1:
                writer.writerow({"step": step + 1, "loss": last_loss,
                                 "xyz_mse": float(xyz_loss.detach().cpu()),
                                 "gripper_bce": float(grip_loss.detach().cpu()),
                                 "seconds": time.monotonic() - start_time})
                stream.flush()
            if (step + 1) % 100 == 0 or step == steps - 1:
                save(step + 1)
    result = {"status": "completed", "steps": steps, "last_loss": last_loss,
              "duration_seconds": time.monotonic() - start_time, "device": device,
              "feature_dim": model_config["feature_dim"], "checkpoint": str(checkpoint_path)}
    (output_dir / "train_metrics.json").write_text(json.dumps(result, indent=2))
    del model, optimizer, features, targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
