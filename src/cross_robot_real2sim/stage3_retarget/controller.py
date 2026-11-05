"""Observable event-based phase tracker shared by teacher collection and BC."""
from __future__ import annotations

import numpy as np

PHASES = ("pregrasp", "grasp", "close", "lift", "transport", "lower", "open", "retreat")
FEATURE_NAMES = [f"{prefix}_{axis}" for prefix in ("eef", "object", "goal", "object_minus_eef", "goal_minus_object") for axis in "xyz"] + ["gripper_opening", "grasp_contact", "object_lift"] + [f"phase_{p}" for p in PHASES]


class PhaseTracker:
    def __init__(self, obs, parameters=None):
        self.parameters = parameters or {"grasp_offset": np.zeros(3), "clearance": .18}
        self.phase = 0
        self.phase_steps = 0
        self.initial_object = np.asarray(obs["object_pos"]).copy()
        self.initial_goal = np.asarray(obs["goal_pos"]).copy()
        self.last_target = None

    def target(self, obs):
        obj, goal = np.asarray(obs["object_pos"]), np.asarray(obs["goal_pos"])
        offset = np.asarray(self.parameters["grasp_offset"])
        clearance = float(self.parameters["clearance"])
        pickup = self.initial_object + offset
        place = np.r_[goal[:2], self.initial_object[2]] + offset
        if self.phase == 0:
            return obj + offset + [0, 0, clearance]
        if self.phase in (1, 2):
            return obj + offset
        if self.phase == 3:
            return pickup + [0, 0, clearance]
        if self.phase == 4:
            return place + [0, 0, clearance]
        if self.phase in (5, 6):
            return place + [0, 0, .015]
        return place + [0, 0, clearance]

    def update(self, obs):
        target = self.target(obs)
        self.phase_steps += 1
        error = float(np.linalg.norm(target - obs["eef_pos"]))
        threshold = .017 if self.phase in (1, 5) else .028
        done = (self.phase in (2, 6) and self.phase_steps >= 14) or (self.phase not in (2, 6) and error < threshold)
        if done and self.phase < 7:
            self.phase += 1
            self.phase_steps = 0
        self.last_target = self.target(obs)
        return self.phase


def build_features(obs, env=None, tracker=None):
    if tracker is None:
        raise ValueError("The same online PhaseTracker is required at collection and inference")
    eef, obj, goal = (np.asarray(obs[k], dtype=float) for k in ("eef_pos", "object_pos", "goal_pos"))
    phase = np.eye(len(PHASES), dtype=float)[tracker.phase]
    result = np.r_[eef, obj, goal, obj-eef, goal-obj, float(obs["gripper_opening"]),
                   float(obs["grasp_contact"]), obj[2]-tracker.initial_object[2], phase].astype(np.float32)
    if result.shape != (26,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Invalid feature schema {result.shape}")
    return result


def teacher_action(obs, tracker, adapter):
    target = tracker.target(obs)
    xyz = adapter.delta_to_normalized((target - obs["eef_pos"]) * .6)
    gripper = 1. if 2 <= tracker.phase <= 5 else -1.
    return np.r_[xyz, gripper].astype(np.float32)
