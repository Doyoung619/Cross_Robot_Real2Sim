"""Render the original Panda motion by restoring every recorded simulator state.

This is state reconstruction, not action playback and not a scripted controller.
The original per-episode MJCF is retained except for resolving public robosuite
asset paths to the installed package. Both EEF and Can positions must match the
dataset at every frame before a result is accepted.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def _sha256(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_asset_root():
    spec = importlib.util.find_spec("robosuite")
    if spec is None or spec.origin is None:
        raise RuntimeError("robosuite is required to resolve the original scene assets")
    return (Path(spec.origin).parent / "models/assets").resolve()


def resolve_original_xml_assets(xml, asset_root=None):
    """Change only file attributes, refusing unknown or escaping asset paths."""
    asset_root = Path(asset_root or installed_asset_root()).resolve()
    root = ET.fromstring(xml)
    rewrites = []
    for element in root.iter():
        source = element.get("file")
        if source is None:
            continue
        marker = "/robosuite/models/assets/"
        if marker not in source:
            raise ValueError(f"Unrecognized original asset path: {source}")
        target = (asset_root / source.split(marker, 1)[1]).resolve()
        if not target.is_relative_to(asset_root):
            raise ValueError(f"Asset path escapes installed robosuite assets: {source}")
        if not target.is_file():
            raise FileNotFoundError(f"Original source asset is unavailable: {target}")
        element.set("file", str(target))
        rewrites.append({"original": source, "resolved": str(target)})
    return ET.tostring(root, encoding="unicode"), rewrites


def load_original_demo(dataset_path, demo_id):
    with h5py.File(dataset_path, "r") as handle:
        group = handle[f"data/{demo_id}"]
        env_args = json.loads(handle["data"].attrs["env_args"])
        xml = group.attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode("utf-8")
        states = np.asarray(group["states"], dtype=np.float64)
        obs = group["obs"]
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        if "Can_pos" in obs:
            can = np.asarray(obs["Can_pos"], dtype=np.float64)
            layout = "named Can_pos"
        else:
            objects = np.asarray(obs["object"], dtype=np.float64)
            if objects.shape != (len(states), 14) or env_args.get("env_name") != "PickPlaceCan":
                raise ValueError("Cannot safely decode the source Can world position")
            can = objects[:, 7:10]
            layout = "robosuite1.5: EEF-relative[0:7], Can world pos[7:10], quat[10:14]"
        actions = np.asarray(group["actions"])
    hz = float(env_args["env_kwargs"]["control_freq"])
    if states.ndim != 2 or eef.shape != (len(states), 3) or can.shape != eef.shape:
        raise ValueError("Malformed source state or observation arrays")
    if len(actions) != len(states) or not all(np.isfinite(x).all() for x in (states, eef, can)):
        raise ValueError("Source lengths or values are invalid")
    expected_times = states[0, 0] + np.arange(len(states)) / hz
    if not np.allclose(states[:, 0], expected_times, atol=1e-8, rtol=0):
        raise ValueError("Original state timestamps do not match declared control frequency")
    return {"demo_id": demo_id, "xml": xml, "states": states, "eef_positions": eef,
            "can_positions": can, "env_args": env_args, "control_hz": hz,
            "object_observation_layout": layout, "action_shape": list(actions.shape)}


def create_original_simulator(demo):
    import mujoco
    resolved_xml, rewrites = resolve_original_xml_assets(demo["xml"])
    model = mujoco.MjModel.from_xml_string(resolved_xml)
    if model.na != 0 or demo["states"].shape[1] != 1 + model.nq + model.nv:
        raise ValueError(f"Original flattened state incompatible with loaded scene: nq={model.nq}, nv={model.nv}, na={model.na}, shape={demo['states'].shape}")
    eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
    can_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Can_main")
    if min(eef_id, can_id) < 0:
        raise ValueError("Original Panda EEF site or Can body is missing from recorded XML")
    return model, mujoco.MjData(model), eef_id, can_id, resolved_xml, rewrites


def restore_original_state(model, data, state):
    """Equivalent to robosuite MjSimState restoration followed by mj_forward."""
    import mujoco
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (1 + model.nq + model.nv,) or not np.isfinite(state).all():
        raise ValueError("Original state shape or values are invalid")
    data.time = state[0]
    data.qpos[:] = state[1:1 + model.nq]
    data.qvel[:] = state[1 + model.nq:]
    mujoco.mj_forward(model, data)


def verify_original_reconstruction(demo, tolerance_m=1e-8):
    model, data, eef_id, can_id, resolved_xml, rewrites = create_original_simulator(demo)
    eef, can = [], []
    for state in demo["states"]:
        restore_original_state(model, data, state)
        eef.append(data.site_xpos[eef_id].copy())
        can.append(data.xpos[can_id].copy())
    eef, can = np.asarray(eef), np.asarray(can)
    eef_error = np.linalg.norm(eef - demo["eef_positions"], axis=1)
    can_error = np.linalg.norm(can - demo["can_positions"], axis=1)
    if max(float(eef_error.max()), float(can_error.max())) > tolerance_m:
        raise ValueError(f"Original source reconstruction mismatch: EEF={eef_error.max():.9g}m Can={can_error.max():.9g}m")
    return {"model": model, "data": data, "eef_site_id": eef_id, "can_body_id": can_id,
            "resolved_xml": resolved_xml, "asset_rewrites": rewrites,
            "reconstructed_eef_positions": eef, "reconstructed_can_positions": can,
            "eef_errors_m": eef_error, "can_errors_m": can_error}


def _english_labels(frame, demo_id, timestamp, index, count, hz):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 21)
    small_font = ImageFont.truetype(font_path, 15)
    draw.rectangle((0, 0, image.width, 64), fill=(12, 21, 34))
    draw.text((12, 8), f"Robot A | Original Panda motion | {demo_id}", font=title_font, fill="white")
    draw.text((12, 37), f"Dataset state reconstruction | Original {hz:g} Hz | Time {timestamp:.2f} s", font=small_font, fill=(184, 218, 244))
    draw.rectangle((0, image.height-31, image.width, image.height), fill=(12, 21, 34))
    draw.text((12, image.height-24), f"Recorded simulator states; no scripted controller.  Frame {index+1}/{count}", font=small_font, fill="white")
    return np.asarray(image)


def render_source_demo(dataset_path, demo_id, output_path, config=None):
    """Return provenance only after all states and rendered frames are verified."""
    config = config or {}
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco
    import imageio.v2 as imageio
    started = time.monotonic()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    demo = load_original_demo(dataset_path, demo_id)
    checked = verify_original_reconstruction(demo, tolerance_m=float(config.get("tolerance_m", 1e-8)))
    model, data = checked["model"], checked["data"]
    width, height = int(config.get("width", 640)), int(config.get("height", 480))
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    camera = config.get("camera", "agentview")
    video_times = np.arange(len(demo["states"])) / demo["control_hz"]
    scratch = Path(config.get("scratch_dir", output.parent))
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch/f"{demo_id}_original_model.xml").write_text(demo["xml"])
    (scratch/f"{demo_id}_resolved_model.xml").write_text(checked["resolved_xml"])
    partial = output.with_name(output.stem+".partial.mp4")
    scene_option=mujoco.MjvOption()
    # Match robosuite's default offscreen rendering: visual geoms visible,
    # collision proxies hidden. State reconstruction itself is unchanged.
    scene_option.geomgroup[0]=0
    scene_option.geomgroup[1]=1
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            with imageio.get_writer(str(partial), fps=demo["control_hz"], codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
                for i, state in enumerate(demo["states"]):
                    restore_original_state(model, data, state)
                    renderer.update_scene(data, camera=camera,scene_option=scene_option)
                    frame = renderer.render().copy()
                    writer.append_data(_english_labels(frame, demo_id, video_times[i], i, len(video_times), demo["control_hz"]))
        # Probe the encoded artifact; only then atomically publish the final file.
        with imageio.get_reader(str(partial)) as reader:
            encoded_frames = reader.count_frames()
            encoded = reader.get_meta_data()
        if encoded_frames != len(demo["states"]) or abs(encoded["fps"]-demo["control_hz"]) > 1e-6:
            raise RuntimeError(f"Encoded output changed original timing/frames: {encoded_frames}, {encoded['fps']}")
        partial.replace(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    validation_path = output.with_name(output.stem+"_validation.npz")
    np.savez_compressed(validation_path, source_sim_timestamps=demo["states"][:, 0],
                        video_timestamps=video_times, source_states=demo["states"],
                        dataset_eef_positions=demo["eef_positions"], dataset_can_positions=demo["can_positions"],
                        reconstructed_eef_positions=checked["reconstructed_eef_positions"],
                        reconstructed_can_positions=checked["reconstructed_can_positions"],
                        eef_errors_m=checked["eef_errors_m"], can_errors_m=checked["can_errors_m"])
    csv_path = output.with_name(output.stem+"_frame_validation.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "source_sim_timestamp_s", "video_timestamp_s", "eef_error_m", "can_error_m"])
        writer.writerows(zip(range(len(video_times)), demo["states"][:, 0], video_times,
                             checked["eef_errors_m"], checked["can_errors_m"]))
    metadata = {"schema_version": "1.0", "status": "completed", "demo_id": demo_id,
                "robot": "Panda", "role": "original_source_robot_A_motion", "mode": "source_dataset_state_reconstruction",
                "source_dataset_path": str(Path(dataset_path).resolve()), "source_dataset_sha256": _file_sha256(dataset_path),
                "source_dataset_id": "robomimic/robomimic_datasets/v1.5/can/ph/low_dim_v15.hdf5",
                "source_policy_type": "recorded_human_demonstration_dataset", "scripted_controller_used": False,
                "action_replay_used": False, "state_reconstruction_used": True,
                "source_env_version": demo["env_args"].get("env_version"),
                "installed_robosuite": importlib.metadata.version("robosuite"), "installed_mujoco": mujoco.__version__,
                "source_model_xml_sha256": _sha256(demo["xml"]), "resolved_model_xml_sha256": _sha256(checked["resolved_xml"]),
                "model_changes": "Only asset file paths resolved; original scene, robot, cameras and states retained",
                "asset_path_rewrites": checked["asset_rewrites"], "camera": camera,
                "render_collision_mesh":False,"render_visual_mesh":True,
                "state_shape": list(demo["states"].shape), "action_shape": demo["action_shape"],
                "object_observation_layout": demo["object_observation_layout"],
                "validated_frames": len(video_times), "encoded_frames": encoded_frames,
                "control_hz": demo["control_hz"], "video_fps": encoded["fps"], "speed_multiplier": 1.0,
                "timestamp_source": "Original simulator state times, checked against metadata control frequency",
                "source_first_timestamp_s": float(demo["states"][0, 0]),
                "source_last_timestamp_s": float(demo["states"][-1, 0]),
                "video_duration_s": len(video_times)/demo["control_hz"],
                "max_eef_position_error_m": float(checked["eef_errors_m"].max()),
                "max_can_position_error_m": float(checked["can_errors_m"].max()),
                "tolerance_m": float(config.get("tolerance_m", 1e-8)),
                "video_path": str(output), "video_sha256": _file_sha256(output), "width": width, "height": height,
                "validation_npz": str(validation_path), "frame_validation_csv": str(csv_path),
                "render_backend": os.environ.get("MUJOCO_GL"), "device": "CPU state reconstruction; GPU 0 EGL rendering",
                "privileged_state_used": True, "human_video_used": False, "labels_language": "English",
                "duration_seconds": time.monotonic()-started,
                "method_reference": "https://robosuite.ai/docs/algorithms/demonstrations.html"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return metadata
