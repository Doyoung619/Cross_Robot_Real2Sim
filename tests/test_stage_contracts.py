from pathlib import Path
import numpy as np
import pytest
from pydantic import ValidationError
from cross_robot_real2sim.common.config import repo_root
from cross_robot_real2sim.common.io import read_json,write_json
from cross_robot_real2sim.schemas import TaskSpec,validate_trajectory_arrays
from cross_robot_real2sim.stage2_task.run import run
from cross_robot_real2sim.stage2_task.qwen_vl import parse_json_object


def test_seed_contract_and_unknown_physics():
    seed=read_json(repo_root()/"data/task_seeds/can_to_bin.json")
    assert TaskSpec.model_validate(seed).task_source=="provided_seed"
    seed["unknowns"]["mass"]="0.1kg"
    with pytest.raises(ValidationError):TaskSpec.model_validate(seed)


def test_unrelated_goal_is_not_coerced():
    seed=read_json(repo_root()/"data/task_seeds/can_to_bin.json")
    seed["goal"]["relation"]="on_top_of"
    with pytest.raises(ValidationError):TaskSpec.model_validate(seed)


def test_full_missing_video_reports_required_input(tmp_path):
    write_json(tmp_path/"01_selection/selection.json",{"selected_human_clip_paths":[]})
    result=run({"mode":"full","seed":42},tmp_path)
    assert result["status"]=="skipped_missing_input"
    assert result["reason"]=="requires_human_video"
    assert not (tmp_path/"02_task/task_spec.json").exists()


def test_bootstrap_uses_explicit_seed(tmp_path):
    write_json(tmp_path/"01_selection/selection.json",{"selected_human_clip_paths":[]})
    result=run({"mode":"bootstrap","seed":42},tmp_path)
    assert result["task_source"]=="provided_seed"
    assert result["goal_pose_source"]=="simulator_oracle"
    assert result["perception_status"]=="skipped_missing_input"


def test_model_json_fence_and_invalid_output():
    assert parse_json_object('```json\n{"object":"can"}\n```')["object"]=="can"
    with pytest.raises(ValueError):parse_json_object("I cannot identify the goal")


def test_array_contract_rejects_misaligned_time_and_action_dimension():
    assert validate_trajectory_arrays(np.zeros((2,10)),np.zeros((2,4)),np.array([0.,.05]))
    with pytest.raises(ValueError):validate_trajectory_arrays(np.zeros((2,10)),np.zeros((2,7)),np.array([0.,.05]))
    with pytest.raises(ValueError):validate_trajectory_arrays(np.zeros((2,10)),np.zeros((2,4)),np.array([0.,0.]))


def test_full_pipeline_without_video_never_trains_or_claims_perception(tmp_path):
    from cross_robot_real2sim.adapters.source_hdf5 import dataset_path
    from cross_robot_real2sim.orchestrator.run_pipeline import run_pipeline
    path=dataset_path()
    if not path.exists():pytest.skip("Minimal Can dataset not installed")
    empty_human=tmp_path/"empty_human";empty_human.mkdir()
    config={"mode":"full","seed":42,"source_demo_limit":2,"selected_source_demo_limit":1,
            "source_dataset_path":str(path),"human_video_dir":str(empty_human),
            "task_seed":str(repo_root()/"data/task_seeds/can_to_bin.json")}
    run_dir=tmp_path/"full_missing_video"
    result=run_pipeline(config,run_dir)
    for step in (2,3,4):
        assert result["stages"][str(step)]["status"]=="skipped_missing_input"
        assert result["stages"][str(step)]["result"]["reason"]=="requires_human_video"
    assert result["status"]!="completed"
    assert not (run_dir/"02_task/task_spec.json").exists()
    assert not (run_dir/"03_retarget/stats.json").exists()
    assert not (run_dir/"04_policy/checkpoint.pt").exists()
    assert (run_dir/"05_diagnosis/feedback_requests.json").exists()
