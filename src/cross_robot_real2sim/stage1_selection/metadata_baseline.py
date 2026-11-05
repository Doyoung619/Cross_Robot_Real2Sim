"""Deterministic source selection from real episode metadata and start diversity."""
import numpy as np
from cross_robot_real2sim.adapters.source_hdf5 import list_demonstrations, extract_trace


def select_demonstrations(path, config):
    candidates=list_demonstrations(path, limit=int(config.get("source_demo_limit",3)))
    candidates=[x for x in candidates if x["consistent_shapes"] and x["source_robot"]==config.get("source_robot","Panda")]
    if not candidates: raise ValueError("No consistent Can/Panda source episodes")
    for candidate in candidates:
        trace=extract_trace(path,candidate["demo_id"])
        candidate["initial_object_xyz"]=trace["object_pos"][0].tolist()
    # First is shortest coherent demo; remaining are farthest starts from the selected set.
    candidates.sort(key=lambda x:(x["num_samples"],x["demo_id"]))
    selected=[candidates[0]]
    remaining=candidates[1:]
    while remaining and len(selected)<int(config.get("selected_source_demo_limit",2)):
        for candidate in remaining:
            candidate["start_diversity_m"]=min(float(np.linalg.norm(np.array(candidate["initial_object_xyz"])-x["initial_object_xyz"])) for x in selected)
        best=max(remaining,key=lambda x:(x["start_diversity_m"],x["demo_id"]))
        selected.append(best);remaining.remove(best)
    return selected,candidates
