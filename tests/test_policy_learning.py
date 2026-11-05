"""CPU contract tests use explicit fixture data, never research rollout evidence."""
import json
import numpy as np
import pytest
import torch
from cross_robot_real2sim.stage4_policy.bc_model import BCPolicy,load_policy
from cross_robot_real2sim.stage4_policy.build_dataset import build_dataset
from cross_robot_real2sim.stage3_retarget.controller import PhaseTracker,build_features,FEATURE_NAMES
from cross_robot_real2sim.stage5_diagnosis.detectors import diagnose_episode


def episode_fixture(path, *, seed=100, success=True, robot="UR5e", origin="source_retargeted", dim=26, value=1.):
    path.parent.mkdir(parents=True,exist_ok=True)
    meta={"seed":seed,"success":success,"robot":robot,"teacher_origin":origin,
          "episode_id":path.stem,"schema_version":"1.0","fixture_only":True}
    np.savez_compressed(path,features=np.full((2,dim),value,dtype=np.float32),
                        actions=np.array([[.1,0,0,1],[0,0,0,-1]],dtype=np.float32),
                        timestamps=np.array([0,.05]),metadata_json=json.dumps(meta))


def test_online_features_have_one_named_contract_and_observable_phase():
    obs={"eef_pos":np.array([0.,0.,1.]),"object_pos":np.array([.1,.2,.86]),
         "goal_pos":np.array([.2,.3,.8]),"gripper_opening":.02,"grasp_contact":False}
    tracker=PhaseTracker(obs)
    features=build_features(obs,tracker=tracker)
    assert features.shape==(len(FEATURE_NAMES),)==(26,)
    np.testing.assert_allclose(features[:3],obs["eef_pos"])
    np.testing.assert_allclose(features[3:6],obs["object_pos"])
    np.testing.assert_allclose(features[6:9],obs["goal_pos"])
    assert features[-8:].sum()==1 and features[-8]==1
    with pytest.raises(ValueError,match="same online PhaseTracker"):
        build_features(obs)
    # The phase changes from online geometry, with no saved expert phase labels.
    obs["eef_pos"]=tracker.target(obs)
    tracker.update(obs)
    assert build_features(obs,tracker=tracker)[-7]==1


def test_successful_training_subset_and_train_only_normalization(tmp_path):
    root=tmp_path/"03_retarget/rollouts"
    episode_fixture(root/"a.npz",seed=100,value=1.)
    episode_fixture(root/"b.npz",seed=101,value=3.)
    episode_fixture(root/"failed.npz",seed=102,success=False,value=999.)
    episode_fixture(root/"source_robot.npz",seed=103,robot="Panda",value=999.)
    episode_fixture(root/"learned.npz",seed=104,origin="learned_policy",value=999.)
    result=build_dataset({"seed":42,"eval_seeds":[10000,10001]},tmp_path)
    assert result["samples"]==4 and len(result["episodes"])==2
    assert {item["seed"] for item in result["episodes"]}=={100,101}
    with np.load(result["path"],allow_pickle=False) as data:
        np.testing.assert_allclose(data["mean"],2.)
        np.testing.assert_allclose(data["std"],1.)
    assert result["normalization_source"]=="training_demonstrations_only"


def test_held_out_seed_leakage_is_rejected(tmp_path):
    episode_fixture(tmp_path/"03_retarget/rollouts/leak.npz",seed=10000)
    with pytest.raises(ValueError,match="seed overlap"):
        build_dataset({"eval_seeds":[10000]},tmp_path)


def test_inconsistent_feature_dimensions_are_rejected(tmp_path):
    episode_fixture(tmp_path/"03_retarget/rollouts/a.npz",seed=100,dim=26)
    episode_fixture(tmp_path/"03_retarget/rollouts/b.npz",seed=101,dim=16)
    with pytest.raises(ValueError,match="Inconsistent feature dimensions"):
        build_dataset({"eval_seeds":[10000]},tmp_path)


def test_zero_successes_block_training_instead_of_synthetic_substitution(tmp_path):
    episode_fixture(tmp_path/"03_retarget/rollouts/failure.npz",success=False)
    result=build_dataset({"eval_seeds":[10000]},tmp_path)
    assert result["status"]=="blocked_insufficient_teacher_data"
    assert result["samples"]==0
    assert not (tmp_path/"04_policy/train_data.npz").exists()


def test_saved_policy_uses_checkpoint_normalization_dimensions_and_action_clipping(tmp_path):
    model=BCPolicy(feature_dim=26,hidden_dim=8)
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.net[0].weight[0,0]=1.
        model.net[2].weight[0,0]=1.
        model.net[4].weight[0,0]=2.
        model.net[4].bias[3]=-1.
    checkpoint={"model_config":{"feature_dim":26,"hidden_dim":8},"model_state_dict":model.state_dict(),
                "normalization":{"mean":[1.]*26,"std":[2.]*26}}
    path=tmp_path/"checkpoint.pt";torch.save(checkpoint,path)
    predict,payload=load_policy(path)
    features=np.ones(26,dtype=np.float32);features[0]=1.4
    np.testing.assert_allclose(predict(features),[.4,0,0,-1],atol=1e-6)
    features[0]=11.
    np.testing.assert_allclose(predict(features),[1,0,0,-1])
    assert payload["model_config"]["feature_dim"]==26
    with pytest.raises(ValueError,match="feature shape"):predict(np.ones(16))
    features[1]=np.nan
    with pytest.raises(ValueError,match="feature shape"):predict(features)


def diagnostic_fixture(path, *, actions=None, positions=None, metadata=None):
    meta={"seed":101,"teacher_origin":"learned_policy","native_task_success":False,
          "success":False,"max_lift_m":.14,"goal_position":[.3,.3,.8],"reason_code":"horizon_timeout"}
    meta.update(metadata or {})
    arrays={"metadata_json":json.dumps(meta)}
    if positions is not None:arrays["object_positions"]=np.asarray(positions,dtype=float)
    if actions is not None:arrays["actions"]=np.asarray(actions,dtype=float)
    arrays["grasp_contacts"]=np.ones(len(actions) if actions is not None else 0,dtype=bool)
    np.savez_compressed(path,**arrays)
    return path


def test_diagnosis_observed_drop_and_native_success_precedence(tmp_path):
    path=diagnostic_fixture(tmp_path/"drop.npz",actions=[[0,0,0,1]]*3,
                            positions=[[0,0,.86],[0,0,1.],[0,0,.86]],
                            metadata={"success":True,"native_task_success":False})
    row=diagnose_episode(path)
    assert not row["success"] and row["reason_code"]=="object_drop"
    assert row["step_attribution"]==4 and row["diagnostic_oracle_used"]
    path=diagnostic_fixture(tmp_path/"success.npz",actions=[[0,0,0,-1]]*3,
                            positions=[[0,0,.86],[.3,.3,.86],[.3,.3,.86]],
                            metadata={"native_task_success":True,"max_lift_m":.14})
    assert diagnose_episode(path)["reason_code"]=="success"


def test_diagnosis_premature_release_from_real_state_rule(tmp_path):
    path=diagnostic_fixture(tmp_path/"release.npz",actions=[[0,0,0,1],[0,0,0,1],[0,0,0,-1]],
                            positions=[[0,0,.86],[0,0,1.],[0,0,1.]])
    row=diagnose_episode(path)
    assert row["reason_code"]=="premature_release" and row["premature_release"]


def test_malformed_action_dimension_is_diagnosed_without_crashing(tmp_path):
    path=diagnostic_fixture(tmp_path/"malformed.npz",actions=[[0,0],[0,0]],positions=[[0,0,.86],[0,0,1.]])
    row=diagnose_episode(path)
    assert row["reason_code"]=="gripper_action_mismatch"


def test_missing_state_evidence_does_not_assert_physical_cause(tmp_path):
    path=diagnostic_fixture(tmp_path/"missing.npz",metadata={"max_lift_m":0.,"reason_code":"unknown"})
    row=diagnose_episode(path)
    assert row["reason_code"]=="unknown" and row["confidence_type"]=="unknown"


def test_checkpoint_optimizer_and_sampling_resume_matches_uninterrupted_cpu_training(tmp_path):
    from shutil import copyfile
    from cross_robot_real2sim.stage4_policy.train import train
    episode_fixture(tmp_path/"resumed/03_retarget/rollouts/a.npz",seed=100,value=1.)
    episode_fixture(tmp_path/"resumed/03_retarget/rollouts/b.npz",seed=101,value=3.)
    build_dataset({"eval_seeds":[10000]},tmp_path/"resumed")
    resumed=tmp_path/"resumed/04_policy"
    uninterrupted=tmp_path/"uninterrupted";uninterrupted.mkdir()
    copyfile(resumed/"train_data.npz",uninterrupted/"train_data.npz")
    config={"device":"cpu","seed":42,"bc_train_steps":2,"bc_hidden_dim":8,"bc_batch_size":2}
    train(config,resumed)
    first=torch.load(resumed/"checkpoint.pt",weights_only=False)
    assert first["step"]==2 and first["optimizer_state_dict"]["state"]
    assert first["dataset_metadata"]["normalization_source"]=="training_demonstrations_only"
    config["bc_train_steps"]=4
    train(config,resumed,resume=True)
    train(config,uninterrupted)
    actual=torch.load(resumed/"checkpoint.pt",weights_only=False)
    expected=torch.load(uninterrupted/"checkpoint.pt",weights_only=False)
    assert actual["step"]==expected["step"]==4
    for key,value in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][key],value,rtol=0,atol=0)
