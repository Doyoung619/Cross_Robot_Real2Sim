import json
import h5py
import numpy as np
import pytest
from cross_robot_real2sim.adapters.source_hdf5 import extract_trace,list_demonstrations


@pytest.fixture
def source_fixture(tmp_path):
    path=tmp_path/"can.hdf5"
    with h5py.File(path,"w") as f:
        data=f.create_group("data")
        data.attrs["env_args"]=json.dumps({"env_name":"PickPlaceCan","env_version":"1.5.1","env_kwargs":{"control_freq":20}})
        demo=data.create_group("demo_0");demo.create_dataset("actions",data=np.zeros((3,7)))
        obs=demo.create_group("obs");obs.create_dataset("robot0_eef_pos",data=np.array([[.1,0,1.0],[.1,0,1.1],[.2,.3,1.1]]))
        state=np.zeros((3,14));state[:,:3]=[9,8,7];state[:,7:10]=[[.1,0,.86],[.1,0,.95],[.2,.3,.86]];state[:,13]=1
        obs.create_dataset("object",data=state)
    return path


def test_object_layout_uses_world_not_eef_relative_fields(source_fixture):
    trace=extract_trace(source_fixture,"demo_0")
    np.testing.assert_allclose(trace["object_pos"][0],[.1,0,.86])
    np.testing.assert_allclose(trace["object_quat"][0],[0,0,0,1])
    np.testing.assert_allclose(trace["timestamps"],[0,.05,.1])
    assert "world_position[7:10]" in trace["metadata"]["object_state_layout"]


def test_inconsistent_hdf5_observation_marked(source_fixture):
    with h5py.File(source_fixture,"a") as f:f["data/demo_0/obs"].create_dataset("bad",data=np.zeros((2,3)))
    assert not list_demonstrations(source_fixture)[0]["consistent_shapes"]


def test_ambiguous_object_layout_rejected(source_fixture):
    with h5py.File(source_fixture,"a") as f:
        del f["data/demo_0/obs/object"]
        f["data/demo_0/obs"].create_dataset("object",data=np.zeros((3,7)))
    with pytest.raises(ValueError,match="Cannot decode"):
        extract_trace(source_fixture,"demo_0")
