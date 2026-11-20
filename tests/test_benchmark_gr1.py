"""Static native-FK contract for the benchmark model; zero target physics steps."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def models(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from cross_robot_real2sim.benchmark.gr1 import ASSET,prepare_asset,RIGHT
    if not ASSET.exists():pytest.skip('Installed official GR1 assets are required')
    from cross_robot_real2sim.benchmark import gr1
    monkeypatch.setattr(gr1,"GENERATED",tmp_path/"robot")
    target=prepare_asset()
    return mujoco.MjModel.from_xml_path(str(ASSET)),mujoco.MjModel.from_xml_string(target.read_text()),RIGHT


def test_fixed_derivative_matches_native_gr1_forward_kinematics(models):
    original,fixed,right=models
    a,b=mujoco.MjData(original),mujoco.MjData(fixed)
    posture={'torso_waist_pitch':.45,'l_shoulder_roll':.1,'l_elbow_pitch':-1.57}
    values=[.1,-.3,.2,-1.2,.1,.2,-.1]
    posture.update(dict(zip(right,values)))
    for name,value in posture.items():
        j=original.joint(name).id;a.qpos[original.jnt_qposadr[j]]=value
    b.qpos[:]=values
    mujoco.mj_forward(original,a);mujoco.mj_forward(fixed,b)
    assert fixed.nq==fixed.nv==fixed.nu==7
    assert {fixed.body(i).name for i in range(1,fixed.nbody)}=={original.body(i).name for i in range(1,original.nbody)}
    for i in range(1,fixed.nbody):
        j=original.body(fixed.body(i).name).id
        np.testing.assert_allclose(b.xpos[i],a.xpos[j],atol=1e-8)
        np.testing.assert_allclose(b.xmat[i],a.xmat[j],atol=1e-8)
    for name in right:
        np.testing.assert_array_equal(original.jnt_range[original.joint(name).id],fixed.jnt_range[fixed.joint(name).id])

