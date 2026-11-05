"""Map an explicit 4-D intermediary to robosuite BASIC/OSC_POSE actions."""
from __future__ import annotations

import numpy as np


class RobotActionAdapter:
    """XYZ are normalized [-1,1] deltas; each unit is 0.05 m/control step.

    The installed OSC controller supplies the scale; orientation deltas are zero
    (hold the downward initial orientation), and positive gripper means close.
    """

    def __init__(self, env):
        self.env = env
        robot = env.robots[0]
        self.controller = robot.composite_controller
        self.arm = robot.arms[0]
        self.part = self.controller.part_controllers[self.arm]
        if self.part.name != "OSC_POSE":
            raise ValueError(f"Expected OSC_POSE, got {self.part.name}")
        self.action_dim = env.action_dim
        self.scale = np.asarray(self.part.output_max[:3], dtype=float)
        if self.action_dim != 7 or self.part.control_dim != 6:
            raise ValueError(f"Unsupported installed action layout: {self.action_dim}, {self.part.control_dim}")

    def to_env(self, intermediary):
        action = np.asarray(intermediary, dtype=float)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("intermediary must contain four finite values")
        action = np.clip(action, -1, 1)
        # Use the robot's actual named-part action vector API, rather than
        # assuming the concatenation is accepted by an arbitrary controller.
        values = {self.arm: np.r_[action[:3], np.zeros(3)], f"{self.arm}_gripper": action[3:]}
        result = np.asarray(self.controller.create_action_vector(values), dtype=float)
        if result.shape != (self.action_dim,):
            raise ValueError(f"Controller created invalid shape {result.shape}")
        return result

    def delta_to_normalized(self, delta):
        return np.clip(np.asarray(delta) / self.scale, -1, 1)

    def description(self):
        return {"name": "BASIC/OSC_POSE", "action_dim": self.action_dim,
                "intermediary_dim": 4, "xyz_delta_unit": "normalized",
                "meters_per_unit": self.scale.tolist(), "orientation": "initial_pose_hold",
                "positive_gripper": "close", "arm": self.arm}
