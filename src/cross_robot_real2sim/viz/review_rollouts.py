"""Re-render frozen experiments with explicit labels and independent placement checks.

These are held-out review replays, never additional training demonstrations.
The original policy/controller continues during the one-second stability check.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from ..common.provenance import file_sha256
from ..adapters.robosuite_env import make_env, observation, object_index, goal_success, render_frame
from ..adapters.robot_action_adapter import RobotActionAdapter
from ..stage3_retarget.controller import PhaseTracker, build_features, teacher_action, PHASES
from ..stage3_retarget.source_trace import load_source_trace
from ..stage3_retarget.waypoints import transfer_parameters

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


@lru_cache(maxsize=16)
def font(size):
    for path in (FONT_PATH, "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def replay_case(root: Path, output: Path, raw: Path, request: dict) -> dict:
    import imageio.v2 as imageio
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

    run = root / "runs" / request["run_id"]
    config_path = run / request.get("config_path", "04_policy/effective_config.yaml")
    config = yaml.safe_load(config_path.read_text())
    config.update(render_width=640, render_height=480)
    checkpoint = root / request["checkpoint"] if request.get("checkpoint") else None
    original_path = root / request["original_npz"] if request.get("original_npz") else None
    source_sha = file_sha256(Path(__file__))
    cache_key = {"request": request, "implementation_sha256": source_sha,
                 "config_sha256": file_sha256(config_path),
                 "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
                 "original_npz_sha256": file_sha256(original_path) if original_path else None}
    meta_path = output / "episodes" / f"{request['case_id']}.json"
    video_path = meta_path.with_suffix(".mp4")
    if meta_path.exists() and video_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing.get("cache_key") == cache_key and existing.get("video_sha256") == file_sha256(video_path):
            print(f"[review] verified existing replay {request['case_id']}", flush=True)
            return existing

    original, original_features = None, None
    if original_path:
        with np.load(original_path, allow_pickle=False) as data:
            original = json.loads(str(data["metadata_json"].item()))
            original_features = np.asarray(data["features"][0])
    origin, robot = request["teacher_origin"], request.get("robot", "UR5e")
    source_trace = None
    if origin == "source_retargeted":
        selection = json.loads((run / "01_selection/selection.json").read_text())
        source_trace = load_source_trace(selection["source_dataset_path"], request["source_demo_id"])
    predict = None
    if checkpoint:
        from ..stage4_policy.bc_model import load_policy
        predict, _ = load_policy(checkpoint, device="cpu")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    (raw / "rollouts").mkdir(parents=True, exist_ok=True)
    env = make_env(config, robot=robot, seed=int(request["seed"]))
    writer = None
    try:
        obs = observation(env)
        adapter = RobotActionAdapter(env)
        params = transfer_parameters(source_trace, origin, clearance=float(config.get("grasp_clearance", .18)))
        if predict and config.get("policy_tracker_parameters"):
            params = dict(config["policy_tracker_parameters"])
            params["grasp_offset"] = np.asarray(params["grasp_offset"])
        tracker = PhaseTracker(obs, params)
        idx = object_index(env)
        goal = obs["goal_pos"].copy()
        half_x, half_y = np.asarray(env.bin_size)[:2] / 4
        corners = np.array([[goal[0]-half_x, goal[1]-half_y, goal[2]+.005],
                            [goal[0]+half_x, goal[1]-half_y, goal[2]+.005],
                            [goal[0]+half_x, goal[1]+half_y, goal[2]+.005],
                            [goal[0]-half_x, goal[1]+half_y, goal[2]+.005]])
        transform = get_camera_transform_matrix(env.sim, "agentview", 480, 640)
        pixels = project_points_from_world_to_camera(corners, transform, 480, 640)
        polygon = [(int(x), int(y)) for y,x in pixels]
        writer = imageio.get_writer(str(video_path), fps=10, codec="libx264", quality=7)
        records = {key: [] for key in ("features", "actions", "object_positions", "eef_positions", "timestamps", "phases", "grasp_contacts")}
        frame_count = 0
        role = {"source_scripted": "Panda 규칙 기반 정책", "source_retargeted": "UR5e 교사: 원본 잡기·들기 정보 + 규칙 제어",
                "learned_policy": "UR5e 학습 정책: 저장된 신경망"}[origin]

        def snapshot():
            return {"native_success": goal_success(env), "specific_can_inside": not env.not_in_bin(obs["object_pos"], idx),
                    "grasp_contact": bool(obs["grasp_contact"]), "object_position": obs["object_pos"].tolist(),
                    "eef_position": obs["eef_pos"].tolist(), "phase": PHASES[tracker.phase],
                    "xy_error_m": float(np.linalg.norm(obs["object_pos"][:2]-goal[:2]))}

        def capture(message, sim_step):
            nonlocal frame_count
            im = Image.fromarray(render_frame(env, config))
            draw = ImageDraw.Draw(im)
            draw.polygon(polygon, outline=(30,255,80), width=3)
            gx,gy = min(x for x,y in polygon), min(y for x,y in polygon)
            draw.text((gx,max(84,gy-22)), "초록색: 캔의 목표 칸", font=font(16), fill=(30,255,80))
            draw.rectangle((0,0,640,79),fill=(8,14,25))
            draw.text((8,3),role,font=font(18),fill="white")
            draw.text((8,29),f"{config['mode']} / seed {request['seed']} / {request.get('variant_label','')}",font=font(16),fill="white")
            draw.text((8,53),"사람 영상 없음 · 정확한 시뮬레이터 좌표 사용",font=font(15),fill=(255,210,110))
            draw.rectangle((0,432,640,480),fill=(8,14,25))
            state=snapshot()
            draw.text((8,435),f"{message} | {sim_step/config['control_hz']:.2f}s | phase={state['phase']}",font=font(15),fill="white")
            draw.text((8,457),f"캔이 목표 칸 안: {state['specific_can_inside']} / 잡기 접촉: {state['grasp_contact']} / XY 오차: {1000*state['xy_error_m']:.1f}mm",font=font(14),fill="white")
            writer.append_data(np.asarray(im))
            frame_count += 1

        capture("실제 시뮬레이터 재실행 시작",0)
        same_initial_features = None
        max_lift, initial_z = 0., float(obs["object_pos"][2])
        def step_controller(step):
            nonlocal obs, same_initial_features, max_lift
            tracker.update(obs)
            features = build_features(obs, env, tracker)
            if step == 0 and original_features is not None:
                same_initial_features = bool(np.allclose(features, original_features, atol=1e-7))
                if not same_initial_features:
                    raise ValueError(f"Original/replay initial feature mismatch: {request['case_id']}")
            action = predict(features) if predict else teacher_action(obs, tracker, adapter)
            records["features"].append(features)
            records["actions"].append(action)
            records["object_positions"].append(obs["object_pos"].copy())
            records["eef_positions"].append(obs["eef_pos"].copy())
            records["timestamps"].append(step/config["control_hz"])
            records["phases"].append(tracker.phase)
            records["grasp_contacts"].append(obs["grasp_contact"])
            env.step(adapter.to_env(action))
            obs=observation(env)
            max_lift=max(max_lift,float(obs["object_pos"][2]-initial_z))

        terminal_step = 0
        for step in range(int(config["horizon"])):
            step_controller(step)
            terminal_step=step+1
            if step%2==0:
                capture("원 실험과 동일한 제어 실행",step+1)
            if goal_success(env) and tracker.phase >= 7:
                break
        terminal = snapshot()
        capture("원래 평가 종료 지점",terminal_step)
        if original is not None and bool(original["success"]) != terminal["native_success"]:
            raise ValueError(f"Original/replay outcome mismatch: {request['case_id']}")
        hold=[]
        if terminal["native_success"]:
            for offset in range(int(config["control_hz"])):
                step_controller(terminal_step+offset)
                hold.append(snapshot())
                if offset%2==0:
                    capture("같은 정책으로 종료 후 1초 추가 검사",terminal_step+offset+1)
        capture("최종 상태",terminal_step+len(hold))
        final=snapshot()
        writer.close();writer=None
        stable=bool(hold and all(row["specific_can_inside"] and not row["grasp_contact"] for row in hold))
        meta={"schema_version":"1.0","case_id":request["case_id"],"source_run_id":request["run_id"],
              "seed":request["seed"],"mode":config["mode"],"teacher_origin":origin,"robot":robot,
              "source_demo_id":request.get("source_demo_id"),"checkpoint":str(checkpoint) if checkpoint else None,
              "source_npz":str(original_path) if original_path else None,
              "source_original_success":original.get("success") if original else None,
              "same_initial_features":same_initial_features,"terminal_step":terminal_step,
              "terminal":terminal,"post_hold":final,"post_hold_steps":len(hold),
              "post_hold_seconds":len(hold)/config["control_hz"],"stable_placement_1s":stable,
              "max_lift_m":max_lift,"goal_position":goal.tolist(),
              "original_terminal_position_delta_m":float(np.linalg.norm(np.asarray(original['final_object_position'])-np.asarray(terminal['object_position']))) if original and original.get('final_object_position') else None,
              "training_performed":False,"usage":"held_out_review_only_never_training",
              "privileged_state_used":True,"observation_mode":"oracle_sim_state",
              "controller":adapter.description(),"hold_controller":"same original policy; no scripted override",
              "video_path":str(video_path.relative_to(output)),"video_frames":frame_count,"fps":10,
              "video_sha256":file_sha256(video_path),"cache_key":cache_key}
        meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n")
        np.savez_compressed(raw/"rollouts"/f"{request['case_id']}.npz",
                            **{key:np.asarray(values) for key,values in records.items()},metadata_json=json.dumps(meta))
        print(f"[review] {request['case_id']} original_success={meta['source_original_success']} replay_success={terminal['native_success']} stable1s={stable}",flush=True)
        return meta
    finally:
        if writer is not None:
            writer.close()
        env.close()
