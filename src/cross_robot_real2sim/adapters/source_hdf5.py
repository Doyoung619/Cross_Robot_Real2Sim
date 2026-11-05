"""Minimal, read-only adapter for the single robomimic Can PH low-dimensional file."""
from pathlib import Path
import argparse
import json
import logging
import numpy as np
import h5py
from cross_robot_real2sim.common.config import resolve_path
from cross_robot_real2sim.common.provenance import file_sha256

DATASET_REPO = "robomimic/robomimic_datasets"
DATASET_FILE = "v1.5/can/ph/low_dim_v15.hdf5"


def dataset_path(config=None):
    config = config or {}
    if config.get("source_dataset_path"):
        return resolve_path(config["source_dataset_path"], config.get("repo_root"))
    return resolve_path(Path("data/source") / config.get("source_dataset", DATASET_FILE), config.get("repo_root"))


def _validate_file(path):
    if not Path(path).is_file():
        return False
    try:
        with h5py.File(path, "r") as f:
            return "data" in f and len(f["data"]) > 0
    except OSError:
        return False


def ensure_dataset(config=None):
    config = config or {}
    path = dataset_path(config)
    if _validate_file(path):
        return path
    if config.get("source_dataset_path"):
        raise ValueError(f"Configured source_dataset_path does not contain valid HDF5: {path}")
    from huggingface_hub import hf_hub_download
    path.parent.mkdir(parents=True, exist_ok=True)
    # A single named file: deliberately never snapshot_download the entire repository.
    downloaded = Path(hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE,
                                     repo_type="dataset", local_dir=str(resolve_path("data/source", config.get("repo_root")))))
    if not _validate_file(downloaded):
        raise ValueError(f"Downloaded HDF5 is invalid: {downloaded}")
    if downloaded.stat().st_size > int(config.get("dataset_size_cap_bytes", 100_000_000)):
        raise ValueError("Source dataset exceeded minimal-file storage budget")
    return downloaded


def _decode(value):
    if isinstance(value, bytes): value = value.decode()
    if isinstance(value, str):
        try: return json.loads(value)
        except (ValueError, TypeError): return value
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def _demo_sort(name):
    try: return int(name.split("_")[-1])
    except ValueError: return name


def list_demonstrations(path, limit=None):
    episodes = []
    with h5py.File(path, "r") as f:
        for name in sorted(f["data"], key=_demo_sort):
            group = f["data"][name]
            if not isinstance(group, h5py.Group) or "actions" not in group or "obs" not in group:
                continue
            n = len(group["actions"])
            keys = {k: list(v.shape) for k,v in group["obs"].items() if isinstance(v, h5py.Dataset)}
            consistent = n > 0 and all(shape[0] == n for shape in keys.values())
            episodes.append({"demo_id": name, "num_samples": n, "consistent_shapes": consistent,
                             "actions_shape": list(group["actions"].shape), "observation_shapes": keys,
                             "source_robot": "Panda", "task": "Can", "dataset_type": "PH",
                             "source_policy_type": "human_demonstration_dataset"})
            if limit is not None and len(episodes) >= int(limit): break
    return episodes


def inspect_dataset(path, demo_limit=3):
    path = Path(path)
    with h5py.File(path, "r") as f:
        metadata = {k:_decode(v) for k,v in f["data"].attrs.items()}
        result = {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                  "sha256": file_sha256(path), "root_keys": list(f.keys()),
                  "data_attrs": metadata, "demo_count": len(f["data"]),
                  "demonstrations": list_demonstrations(path, demo_limit)}
    return result


def extract_trace(path, demo_id, control_hz=20):
    """Return source arrays with only explicitly available fields.

    PickPlaceCan v1.5 single-object object-state is [relative_pos(3), relative_quat(4), Can_pos(3), Can_quat(4)].
    Prefer named object fields; for object-state, verify env_args names this task.
    Source: robosuite v1.5 PickPlace._setup_observables, ordered EEF-relative position/quaternion, then world position/quaternion.
    Timestamps are reconstructed from documented control frequency, not measured video time.
    """
    with h5py.File(path,"r") as f:
        g=f["data"][demo_id]
        obs=g["obs"]
        env_args=_decode(f["data"].attrs.get("env_args", "{}"))
        if not isinstance(env_args,dict): env_args={}
        if "robot0_eef_pos" not in obs:
            raise KeyError(f"Source has no robot0_eef_pos; actual keys={list(obs)}")
        eef=np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        available = list(obs)
        object_key=next((k for k in ("Can_pos", "can_pos", "object_pos") if k in obs), None)
        object_quat=None
        if object_key:
            obj=np.asarray(obs[object_key], dtype=np.float64)
            qkey=object_key.replace("pos", "quat")
            if qkey in obs: object_quat=np.asarray(obs[qkey],dtype=np.float64)
            object_layout=f"named observation {object_key}"
        elif "object" in obs:
            if "Can" not in str(env_args.get("env_name", "")):
                raise ValueError("Refusing to decode unnamed object state without PickPlaceCan metadata")
            state=np.asarray(obs["object"],dtype=np.float64)
            if state.ndim!=2 or state.shape[1]!=14:
                raise ValueError(f"Cannot decode object state shape {state.shape}")
            obj=state[:,7:10]
            object_quat=state[:,10:14]
            object_layout="PickPlaceCan v1.5 single-object object-state: EEF-relative[0:7], world_position[7:10], world_quaternion_xyzw[10:14]"
        else:
            raise KeyError(f"No usable object position field; actual keys={available}")
        actions=np.asarray(g["actions"],dtype=np.float64)
        hz=float(env_args.get("env_kwargs",{}).get("control_freq", control_hz))
        result={"eef_pos":eef,"object_pos":obj,"actions":actions,
                "timestamps":np.arange(len(eef),dtype=np.float64)/hz,
                "metadata":{"demo_id":demo_id,"source_dataset_path":str(Path(path).resolve()),
                            "source_robot":"Panda", "source_frame":"source_world",
                            "object_state_layout":object_layout,"available_observation_keys":available,
                            "timestamp_source":"reconstructed_from_control_frequency", "control_hz":hz,
                            "source_policy_type":"human_demonstration_dataset", "privileged_state_used":True}}
        if "robot0_eef_quat" in obs: result["eef_quat"]=np.asarray(obs["robot0_eef_quat"],dtype=np.float64)
        if object_quat is not None: result["object_quat"]=object_quat
        if "robot0_gripper_qpos" in obs: result["gripper_qpos"]=np.asarray(obs["robot0_gripper_qpos"],dtype=np.float64)
    for key in ("eef_pos", "object_pos"):
        if result[key].shape != (len(actions),3) or not np.isfinite(result[key]).all():
            raise ValueError(f"Invalid {key} shape/values {result[key].shape}")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--inspect",action="store_true")
    parser.add_argument("--path")
    args=parser.parse_args()
    path=ensure_dataset() if args.download else Path(args.path) if args.path else dataset_path()
    print(json.dumps(inspect_dataset(path) if args.inspect else {"dataset_path":str(path)},indent=2))

if __name__=="__main__": main()
