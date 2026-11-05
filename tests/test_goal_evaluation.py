import numpy as np
import pytest
from cross_robot_real2sim.adapters.robosuite_env import make_env,observation,goal_success,object_index


@pytest.mark.sim
def test_actual_task_goal_requires_correct_bin_and_release_distance():
    env=make_env({"horizon":10},seed=42)
    try:
        obs=observation(env)
        assert not goal_success(env)
        obj=env.objects[object_index(env)]
        goal=obs["goal_pos"].copy();goal[2]+=.06
        q=env.sim.data.get_joint_qpos(obj.joints[0]).copy()
        q[:3]=goal
        env.sim.data.set_joint_qpos(obj.joints[0],q);env.sim.forward()
        assert goal_success(env)
        q[1]-=.5
        env.sim.data.set_joint_qpos(obj.joints[0],q);env.sim.forward()
        assert not goal_success(env)
    finally: env.close()
