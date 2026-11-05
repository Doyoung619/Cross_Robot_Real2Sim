"""Derive object-relative grasp and lift geometry from an actual demonstration."""
import numpy as np
from ..adapters.source_hdf5 import extract_trace


def load_source_trace(path, demo_id, control_hz=20):
    trace = extract_trace(path, demo_id, control_hz=control_hz)
    eef = np.asarray(trace["eef_pos"])
    obj = np.asarray(trace["object_pos"])
    relative = eef - obj
    # Source closest approach while gripper is closing; fallback to geometric
    # closest approach is explicitly recorded when gripper fields unavailable.
    actions = np.asarray(trace["actions"])
    closing = np.flatnonzero(actions[:, -1] > 0)
    candidates = closing if len(closing) else np.arange(len(eef))
    distances = np.linalg.norm(relative[candidates], axis=1)
    grasp_idx = int(candidates[np.argmin(distances)])
    offset = relative[grasp_idx].copy()
    if np.linalg.norm(offset) > .15 or not (.5 < float(obj[0, 2]) < 1.5):
        raise ValueError("Source grasp/object geometry is implausible; inspect actual HDF5 layout")
    max_lift = float(np.max(obj[:, 2]) - obj[0, 2])
    trace["relative_eef"] = relative
    trace["grasp_index"] = np.array(grasp_idx)
    trace["grasp_offset"] = offset
    trace["lift_clearance"] = np.array(max_lift)
    trace["metadata"].update({"schema_version": "1.0", "grasp_index": grasp_idx, "grasp_offset_m": offset.tolist(),
                              "object_relative_definition": "translation-relative; axes aligned to simulator world; cylindrical can",
                              "source_lift_m": max_lift, "derivation": "source_closing_closest_approach" if len(closing) else "source_geometric_closest_approach"})
    return trace
