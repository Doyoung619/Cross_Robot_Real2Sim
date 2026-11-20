# Frozen compatibility snapshot of the local GR1 runtime, kept separate so that
# the benchmark does not require committing or changing pre-existing work.
# Derived from scripts/vla_humanoid_runtime.py (local 2026-11-14 implementation).
"""Fixed-body GR1 right-arm target in the isolated LIBERO 1.4 runtime.

The official installed GR1 mesh/body/arm model is retained. Non-right-arm
joints are baked into fixed poses and removed, rather than reset every step.
The right tool is Robotiq85, not the native Fourier dexterous hand.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from vla_pilot_common import ROOT, MATERIALS, write_json

ASSET = ROOT / ".venv/lib/python3.10/site-packages/robosuite/models/assets/robots/gr1/robot.xml"
GENERATED = ROOT / "runs/benchmark_shared/gr1_model"
RIGHT = ("r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow_pitch",
         "r_wrist_yaw", "r_wrist_roll", "r_wrist_pitch")
BASE_POSE = np.array([-.53, .23, .95])


def prepare_asset():
    GENERATED.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(ASSET); root = tree.getroot()
    frozen = []
    fixed_q = {"l_shoulder_roll": .1, "l_elbow_pitch": -1.57, "torso_waist_pitch": .45}
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "freejoint":
                parent.remove(child); frozen.append("floating_base")
            elif child.tag == "joint" and parent.tag == "body" and child.get("name") not in RIGHT:
                name = child.get("name"); angle = fixed_q.get(name, 0.)
                if angle:
                    axis = np.fromstring(child.get("axis", "0 0 1"), sep=" ")
                    r0 = Rotation.from_quat(np.fromstring(parent.get("quat", "1 0 0 0"), sep=" ")[[1,2,3,0]])
                    rj = Rotation.from_rotvec(axis * angle)
                    pivot = np.fromstring(child.get("pos", "0 0 0"), sep=" ")
                    pos = np.fromstring(parent.get("pos", "0 0 0"), sep=" ") + r0.apply(pivot-rj.apply(pivot))
                    quat = (r0*rj).as_quat()[[3,0,1,2]]
                    parent.set("pos", " ".join(map(str,pos))); parent.set("quat", " ".join(map(str,quat)))
                parent.remove(child); frozen.append(name)
    # The robot's example floor is replaced by the LIBERO arena floor.
    wb = root.find("worldbody")
    for node in list(wb):
        if node.tag != "body": wb.remove(node)
    eye=np.array([-1.65,-1.8,1.25]);look=np.array([-.35,.08,.95]);forward=look-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    quat=Rotation.from_matrix(np.column_stack([right,up,-forward])).as_quat()[[3,0,1,2]]
    ET.SubElement(wb,"camera",name="humanoid_overview",pos=" ".join(map(str,eye)),quat=" ".join(map(str,quat)),fovy="55")
    for n in list(root.find("actuator")):
        if n.get("joint") not in RIGHT: root.find("actuator").remove(n)
    for node in root.find("asset"):
        if node.get("file"):
            node.set("file", str((ASSET.parent/node.get("file")).resolve()))
    for camera in root.iter("camera"):
        if camera.get("name") == "eye_in_right_hand": camera.set("name", "eye_in_hand")
    # Explicit names make collision provenance stable across compilation.
    for i, geom in enumerate(root.iter("geom")):
        if not geom.get("name"): geom.set("name", f"gr1_geom_{i}")
    # Names are legal on worldbody geoms, but not on MJCF default templates.
    for default in root.findall('.//default'):
        for geom in default.iter('geom'):geom.attrib.pop('name',None)
    target = GENERATED / "robot.xml"
    tree.write(target, encoding="unicode")
    meta = dict(source_asset=str(ASSET.relative_to(ROOT)), source_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
                generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                movable_arm_joints=list(RIGHT), fixed_removed_joints=frozen,
                fixed_qpos=fixed_q, target_tool="Robotiq85Gripper",
                interpretation="GR1 full body, fixed torso/head/legs/left arm, native right arm, replacement parallel gripper",
                source_reference="https://github.com/ARISE-Initiative/robosuite/blob/v1.5.2/robosuite/models/robots/manipulators/gr1_robot.py")
    write_json(GENERATED/"model_provenance.json", meta)
    return target


def register():
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm
    path = prepare_asset()
    if "MountedGR1FixedRightArm" not in REGISTERED_ROBOTS:
        class MountedGR1FixedRightArm(ManipulatorModel):
            def __init__(self,idn=0): super().__init__(str(path),idn=idn)
            @property
            def default_mount(self): return None
            @property
            def default_gripper(self): return "Robotiq85Gripper"
            @property
            def default_controller_config(self): return "default_panda"
            @property
            def init_qpos(self): return np.array([0.,-.1,0.,-1.57,0.,0.,0.])
            @property
            def base_xpos_offset(self): return {k:(lambda length: BASE_POSE.copy()) for k in ("table","kitchen_table","study_table")}
            @property
            def top_offset(self): return np.array([0,0,1.])
            @property
            def _horizontal_radius(self): return .6
            @property
            def arm_type(self): return "single"
            @property
            def _eef_name(self): return "right_mount"
        ROBOT_CLASS_MAPPING["MountedGR1FixedRightArm"] = SingleArm


def make_env(robot="GR1FixedRightArm",size=256):
    import libero.libero as settings
    settings._assets_path_cache = str(MATERIALS/"assets")
    register()
    from libero.libero.envs import OffScreenRenderEnv
    path = MATERIALS/"hf_libero-0.1.3/libero/libero/bddl_files/libero_goal/push_the_plate_to_the_front_of_the_stove.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(path),robots=["GR1FixedRightArm"],
        camera_heights=size,camera_widths=size,control_freq=20,horizon=300,ignore_done=True,
        hard_reset=False,render_gpu_device_id=0)
    env.reset()
    env._fixed_reference=fixed_body_snapshot(env)
    env._max_fixed_position_drift=0.;env._max_fixed_quaternion_drift=0.;env._fixed_guard_steps=0
    original_step=env.step
    def checked_step(action):
        value=original_step(action)
        _,p,q=fixed_body_snapshot(env);_,p0,q0=env._fixed_reference
        env._max_fixed_position_drift=max(env._max_fixed_position_drift,float(np.max(np.abs(p-p0))))
        env._max_fixed_quaternion_drift=max(env._max_fixed_quaternion_drift,float(np.max(np.abs(q-q0))))
        env._fixed_guard_steps+=1
        if max(env._max_fixed_position_drift,env._max_fixed_quaternion_drift)>1e-8:raise RuntimeError("Frozen humanoid body moved")
        return value
    env.step=checked_step
    return env


def fixed_body_snapshot(env):
    from vla_pilot_processing import raw_target
    model,data=raw_target(env)
    import mujoco
    ids=[]
    for i in range(model.nbody):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or ""
        if not name.startswith("robot0_"):continue
        moving=False;j=i
        while j>0:
            if model.body_jntnum[j]>0:moving=True;break
            j=int(model.body_parentid[j])
        if not moving:ids.append(i)
    return ids, data.xpos[ids].copy(), data.xquat[ids].copy()


def frame(env,condition,demo,step,success,distance):
    from PIL import Image,ImageDraw,ImageFont
    image=Image.new("RGB",(1280,800),(12,20,32));draw=ImageDraw.Draw(image)
    path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title=ImageFont.truetype(path,26);body=ImageFont.truetype(path,21)
    draw.text((22,18),"Panda -> fixed-body GR1 | "+condition,font=title,fill="white")
    draw.text((22,62),"Task: push the plate to the front of the stove",font=body,fill=(190,220,245))
    draw.text((22,108),"Full GR1: torso, head, legs and left arm welded; right arm + Robotiq85 active",font=body,fill="white")
    for camera,x,y,size in [("robot0_humanoid_overview",20,180,512),("agentview",580,200,256),("robot0_eye_in_hand",940,200,256)]:
        raw=env.sim.render(width=size,height=size,camera_name=camera)
        image.paste(Image.fromarray(raw[::-1].copy()),(x,y))
    draw.text((580,160),"External camera",font=body,fill="white");draw.text((940,160),"Wrist camera",font=body,fill="white")
    draw.text((570,510),f"{demo} | time {step/20:.2f}s",font=body,fill="white")
    draw.text((570,555),f"Native success: {success} | XY error: {distance*100:.2f} cm",font=body,fill=(110,230,170) if success else (255,195,140))
    draw.text((570,602),f"Frozen-body drift: {env._max_fixed_position_drift:.2e} m",font=body,fill="white")
    draw.text((22,738),"Live VLA-only physics rollout. No scripted pushing or target action supervision.",font=body,fill=(190,220,245))
    return np.asarray(image)


