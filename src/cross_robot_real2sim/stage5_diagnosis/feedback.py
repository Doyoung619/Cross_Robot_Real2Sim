from __future__ import annotations

from collections import Counter


def make_feedback(rows: list[dict], config: dict) -> dict:
    counts = Counter(row["reason_code"] for row in rows if not row["success"])
    requests = {
        "grasp_failed": "Collect successful target teacher starts around the observed grasp failures; inspect grasp offset and alignment.",
        "insufficient_lift": "Increase bounded grasp clearance and inspect whether contact persists during lift.",
        "object_drop": "Add successful transport demonstrations and preserve gripper closure until the can reaches the bin.",
        "premature_release": "Add target transport/lower demonstrations and inspect gripper loss weighting.",
        "unreachable_waypoint": "Check workspace limits and source object-relative offsets; use reachable target starts.",
        "horizon_timeout": "Add executed state coverage and increase bounded BC optimization; inspect online phase transitions.",
        "unknown": "Inspect saved object/EEF states and task mapping before assigning a physical cause.",
        "gripper_action_mismatch": "Repair the installed robot action adapter before collecting more data.",
    }
    ranked = [{"reason_code": code, "count": count, "next_experiment": requests.get(code, requests["unknown"])}
              for code,count in counts.most_common(3)]
    return {"schema_version": "1.0", "seed": config.get("seed"), "mode": config.get("mode"),
            "model_name": "deterministic_simulator_diagnostics", "diagnostic_oracle_used": True,
            "input_ids": [row["rollout_path"] for row in rows], "ranked_bottlenecks": ranked,
            "requests": [{"reason_code": item["reason_code"], "request": item["next_experiment"],
                          "automatic_intervention": "bounded_additional_teacher_collection_and_BC_retraining"}
                         for item in ranked],
            "unmeasured": ["real-world friction", "real-world mass", "collision attribution unless explicitly logged",
                           "human-video interpretation accuracy without a real human clip"]}
