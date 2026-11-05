"""Check actual frozen-run media and its provenance, without retraining."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/review_20261105"
pytestmark=pytest.mark.integration


@pytest.fixture
def review():
    path=OUT/"replay_manifest.json"
    if not path.exists():pytest.skip("Run scripts/launch_review_media.sh first")
    return json.loads(path.read_text())


def test_all_original_fast_successes_are_present_and_stable(review):
    original=list(csv.DictReader((ROOT/"results/fast_20261105T182127Z/04_policy/eval.csv").open()))
    success_seeds={int(row["seed"]) for row in original if row["success"]=="True"}
    replays={case["seed"]:case for case in review["replays"] if case["case_id"].startswith("fast_bc_before_")}
    assert set(replays)=={int(row["seed"]) for row in original}
    assert {seed for seed,case in replays.items() if case["terminal"]["native_success"]}==success_seeds
    for seed in success_seeds:
        case=replays[seed]
        assert case["source_original_success"] and case["same_initial_features"]
        assert case["stable_placement_1s"] and case["post_hold_seconds"]==1.0
        assert case["hold_controller"]=="same original policy; no scripted override"
        assert not case["training_performed"]
    assert review["no_new_training"] and not review["human_video_used"]


def test_frozen_inputs_unchanged_and_bootstrap_feedback_case_real(review):
    by_id={case["case_id"]:case for case in review["replays"]}
    for case in review["replays"]:
        if case.get("source_npz") and Path(case["source_npz"]).exists():
            assert hashlib.sha256(Path(case["source_npz"]).read_bytes()).hexdigest()==case["cache_key"]["original_npz_sha256"]
        if case.get("checkpoint") and Path(case["checkpoint"]).exists():
            assert hashlib.sha256(Path(case["checkpoint"]).read_bytes()).hexdigest()==case["cache_key"]["checkpoint_sha256"]
    before=by_id["bootstrap_bc_before_10002"]
    after=by_id["bootstrap_bc_after_10002"]
    assert before["seed"]==after["seed"] and before["same_initial_features"] and after["same_initial_features"]
    assert not before["terminal"]["native_success"] and after["terminal"]["native_success"]
    assert after["stable_placement_1s"]


def test_media_hashes_and_stage_coverage_are_complete():
    path=OUT/"media_manifest.json"
    if not path.exists():pytest.skip("Review movies are not assembled")
    manifest=json.loads(path.read_text())
    assert len(manifest["movies"])==4
    assert len(manifest["verified_media"])==22
    for item in manifest["verified_media"]:
        video=OUT/item["path"]
        assert video.stat().st_size==item["bytes"]
        assert hashlib.sha256(video.read_bytes()).hexdigest()==item["sha256"]
        assert item["frames"]>0 and item["duration_seconds"]>0
    for movie in manifest["movies"][:2]:
        titles=[chapter["title"] for chapter in movie["chapters"]]
        for stage in range(1,6):assert any(title.startswith(f"{stage}단계") for title in titles)
    assert manifest["teacher_vs_failed_policy_initial_features_match"]


def test_teacher_failure_comparison_uses_same_initial_condition(review):
    by_id={case["case_id"]:case for case in review["replays"]}
    teacher=by_id["fast_teacher_10000"]
    policy=by_id["fast_bc_before_10000"]
    assert teacher["seed"]==policy["seed"]
    assert teacher["terminal"]["native_success"] and not policy["terminal"]["native_success"]
    raw=ROOT/"runs/review_media_20261105_success/rollouts"
    if raw.exists():
        with np.load(raw/"fast_teacher_10000.npz") as a,np.load(raw/"fast_bc_before_10000.npz") as b:
            np.testing.assert_allclose(a["features"][0],b["features"][0],atol=1e-7)
