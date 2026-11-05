import numpy as np
import pytest
from cross_robot_real2sim.adapters.robosuite_env import make_env, observation, smoke_test


@pytest.mark.sim
def test_panda_ur5e_three_step_headless_render(tmp_path):
    result=smoke_test({"horizon":10},tmp_path)
    assert set(result["robots"])=={"Panda","UR5e"}
    for robot in result["robots"].values():
        assert robot["steps"]==3 and robot["rendered_frames"]==3
        assert robot["action_dim"]==7
        assert robot["frame_shape"]==[224,224,3]


@pytest.mark.sim
def test_scene_seed_reset_determinism_and_seed_variation():
    positions=[]
    for seed in [97,97,98]:
        env=make_env({"horizon":10},seed=seed)
        try: positions.append(observation(env)["object_pos"])
        finally: env.close()
    assert np.allclose(positions[0],positions[1],atol=1e-9)
    assert not np.allclose(positions[0],positions[2],atol=1e-5)
