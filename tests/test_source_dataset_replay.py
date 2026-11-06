from pathlib import Path

import numpy as np
import pytest

from cross_robot_real2sim.viz.source_dataset_replay import (
    load_original_demo, resolve_original_xml_assets, verify_original_reconstruction,
)


def test_original_asset_resolution_changes_paths_only_and_rejects_escape(tmp_path):
    (tmp_path/"texture.png").write_bytes(b"test")
    xml='<mujoco><asset><texture name="original" file="/old/robosuite/models/assets/texture.png"/></asset><worldbody><body name="robot" pos="1 2 3"/></worldbody></mujoco>'
    resolved, changes=resolve_original_xml_assets(xml,tmp_path)
    assert str(tmp_path/"texture.png") in resolved
    assert 'pos="1 2 3"' in resolved and 'name="original"' in resolved
    assert len(changes)==1
    with pytest.raises(ValueError,match="escapes"):
        resolve_original_xml_assets('<mujoco><asset><texture file="/old/robosuite/models/assets/../secret"/></asset></mujoco>',tmp_path)
    with pytest.raises(ValueError,match="Unrecognized"):
        resolve_original_xml_assets('<mujoco><asset><texture file="/etc/passwd"/></asset></mujoco>',tmp_path)


@pytest.mark.sim
@pytest.mark.parametrize("demo_id",["demo_3","demo_5","demo_7"])
def test_every_original_state_matches_original_can_eef_and_timestamps(demo_id):
    dataset=Path(__file__).resolve().parents[1]/"data/source/v1.5/can/ph/low_dim_v15.hdf5"
    if not dataset.exists():
        pytest.skip("Public source dataset must be downloaded by setup.sh")
    demo=load_original_demo(dataset,demo_id)
    verified=verify_original_reconstruction(demo)
    assert demo["control_hz"]==20
    assert np.allclose(np.diff(demo["states"][:,0]),.05,rtol=0,atol=1e-8)
    assert np.max(verified["eef_errors_m"])<1e-12
    assert np.max(verified["can_errors_m"])<1e-12
    assert len(verified["reconstructed_eef_positions"])==len(demo["states"])
    # Independently reject mismatched source observations, rather than merely
    # accepting that a renderer produced a plausible moving robot.
    altered=dict(demo);altered["eef_positions"]=demo["eef_positions"]+np.array([.001,0,0])
    with pytest.raises(ValueError,match="reconstruction mismatch"):
        verify_original_reconstruction(altered)
