import numpy as np
import pytest
from cross_robot_real2sim.stage3_retarget.object_frame import to_object_relative, to_target_world
from cross_robot_real2sim.adapters.robot_action_adapter import RobotActionAdapter


def test_frame_transform_reconstruction_and_translation_equivariance():
    eef=np.array([[.4,.2,.9],[.5,.3,1.1]])
    objects=np.array([[.3,.1,.8],[.4,.2,1.]])
    offsets=to_object_relative(eef,objects)
    assert np.allclose(to_target_world(offsets,objects),eef)
    shift=np.array([1.,-.5,.3])
    assert np.allclose(to_object_relative(eef+shift,objects+shift),offsets)
    with pytest.raises(ValueError): to_object_relative(np.zeros((2,3)),np.zeros((3,3)))


@pytest.mark.sim
def test_adapter_uses_installed_seven_dimensional_layout_and_clips():
    from cross_robot_real2sim.adapters.robosuite_env import make_env
    env=make_env({"horizon":10},seed=42)
    try:
        adapter=RobotActionAdapter(env)
        action=adapter.to_env([8.,-4.,.3,2.])
        assert action.shape==(7,)
        assert np.allclose(action,[1.,-1.,.3,0.,0.,0.,1.])
        with pytest.raises(ValueError): adapter.to_env([0.,1.])
        with pytest.raises(ValueError): adapter.to_env([0.,0.,float('nan'),0.])
    finally: env.close()
