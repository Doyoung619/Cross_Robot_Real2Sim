"""Small task-space policy; inference shares the exact collection feature contract."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BCPolicy(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4))

    def forward(self, features):
        return self.net(features)


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = BCPolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = torch.as_tensor(checkpoint["normalization"]["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(checkpoint["normalization"]["std"], dtype=torch.float32, device=device)

    @torch.inference_mode()
    def predict(features):
        array = np.asarray(features, dtype=np.float32)
        if array.shape != (len(mean),) or not np.isfinite(array).all():
            raise ValueError(f"Invalid online feature shape {array.shape}; expected {(len(mean),)}")
        value = model((torch.as_tensor(array, device=device) - mean) / std).cpu().numpy()
        return np.concatenate([np.clip(value[:3], -1, 1), [1.0 if value[3] >= 0 else -1.0]]).astype(np.float32)

    return predict, checkpoint
