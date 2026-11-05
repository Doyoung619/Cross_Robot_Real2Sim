import json
from pathlib import Path
import pytest
from cross_robot_real2sim.orchestrator.cache import artifact_index,valid_stage


def test_stage_cache_detects_tampering_missing_extra_and_video_outputs(tmp_path):
    stage = tmp_path/"03_retarget"
    stage.mkdir()
    artifact=stage/"stats.json"
    artifact.write_text('{"schema_version":"1.0","status":"completed"}')
    videos=tmp_path/"videos"
    videos.mkdir()
    video=videos/"source.mp4"
    video.write_bytes(b"explicit fixture bytes for hash verification")
    status={"status":"completed","input_fingerprint":"fixture_input_hash","artifacts":artifact_index(stage,tmp_path)}
    path=stage/"status.json"
    path.write_text(json.dumps(status))
    assert valid_stage(path,tmp_path,"fixture_input_hash")
    assert not valid_stage(path,tmp_path,"changed_input_hash")
    original=artifact.read_text()
    artifact.write_text(original.replace("completed","cancelled"))
    assert not valid_stage(path,tmp_path,"fixture_input_hash")
    artifact.write_text(original)
    extra=stage/"unindexed_episode.json"
    extra.write_text("{}")
    assert not valid_stage(path,tmp_path,"fixture_input_hash")
    extra.unlink()
    assert valid_stage(path,tmp_path,"fixture_input_hash")
    video.unlink()
    assert not valid_stage(path,tmp_path,"fixture_input_hash")


def test_corrupt_torch_checkpoint_invalidates_cache(tmp_path):
    stage=tmp_path/"04_policy"
    stage.mkdir()
    (stage/"checkpoint.pt").write_bytes(b"deliberately corrupt fixture")
    path=stage/"status.json"
    path.write_text(json.dumps({"status":"completed","input_fingerprint":"test","artifacts":artifact_index(stage,tmp_path)}))
    assert not valid_stage(path,tmp_path,"test")
