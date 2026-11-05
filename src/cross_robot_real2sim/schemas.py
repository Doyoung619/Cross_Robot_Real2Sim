"""Validated JSON contracts. Trajectory arrays are checked at their producer boundary."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np

SCHEMA_VERSION = "1.0"
StageStatus = Literal["completed", "completed_degraded", "failed", "skipped_missing_input"]

class Goal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relation: Literal["inside"]
    receptacle: Literal["bin"]

class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: Literal["1.0"] = "1.0"
    task_id: str = "can_to_bin"
    object: Literal["can"]
    goal: Goal
    stages: list[str] = Field(min_length=3)
    constraints: list[str]
    unknowns: dict[str, str]
    task_source: Literal["provided_seed", "human_video_qwen"]
    goal_pose_source: Literal["simulator_oracle"]
    semantic_to_simulator: dict[str, str]
    assumptions: list[str]

    @field_validator("unknowns")
    @classmethod
    def preserve_unknown_physics(cls, value):
        for key in ("mass", "friction", "contact_state"):
            if value.get(key) != "unknown":
                raise ValueError(f"{key} must remain unknown: monocular RGB/provided task does not measure physics")
        return value

class Selection(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: Literal["1.0"] = "1.0"
    selected_human_clip_paths: list[str]
    selected_source_demonstration_ids: list[str] = Field(min_length=1)
    source_dataset_path: str
    selection_strategy: str
    human_clip_selection: str


def validate_trajectory_arrays(features, actions, timestamps):
    features, actions, timestamps = map(np.asarray, (features, actions, timestamps))
    if features.ndim != 2 or actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError("Expected features [T,D] and task actions [T,4]")
    if len(features) != len(actions) or timestamps.shape != (len(actions),):
        raise ValueError("Trajectory lengths are inconsistent")
    if not all(np.isfinite(x).all() for x in (features, actions, timestamps)):
        raise ValueError("Trajectory contains non-finite values")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must increase")
    return True
