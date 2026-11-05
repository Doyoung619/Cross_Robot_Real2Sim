"""Small source-relative target waypoints, with explicit embodiment calibration."""
import numpy as np


def transfer_parameters(source_trace=None, origin="target_scripted_bootstrap", clearance=0.16):
    if origin == "source_retargeted":
        if source_trace is None:
            raise ValueError("source_retargeted requires measured source trace")
        raw = np.asarray(source_trace["grasp_offset"], dtype=float)
        # Robomimic EEF and target grip-site are both grip-center sites. Keep XY
        # offsets, limiting only extreme source tilt to target parallel jaws.
        offset = np.r_[np.clip(raw[:2], -.025, .025), np.clip(raw[2], -.015, .04)]
        lift = float(np.clip(source_trace["lift_clearance"], .12, .25))
        return {"grasp_offset": offset, "clearance": max(clearance, lift),
                "source_raw_grasp_offset": raw.tolist(), "source_lift_clearance": float(source_trace["lift_clearance"]),
                "embodiment_calibration": "grip_site_xy_clip_2.5cm_z_clip_-1.5_to_4cm"}
    return {"grasp_offset": np.array([0.,0.,0.]), "clearance": float(clearance),
            "embodiment_calibration": "target_scripted_grip_center"}
