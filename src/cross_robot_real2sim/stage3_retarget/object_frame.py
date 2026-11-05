import numpy as np


def to_object_relative(points, object_positions):
    points, objects = np.asarray(points), np.asarray(object_positions)
    if points.shape != objects.shape or points.shape[-1] != 3:
        raise ValueError("Matched (...,3) world positions required")
    return points - objects


def to_target_world(offsets, target_position):
    return np.asarray(offsets) + np.asarray(target_position)
