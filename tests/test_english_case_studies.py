"""Audit actual English A/source -> B/teacher -> shared-policy evidence."""
import csv
import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/english_case_studies_20261106"
pytestmark=pytest.mark.integration


@pytest.fixture
def manifest():
    path=OUT/"case_manifest.json"
    if not path.exists():pytest.skip("Build English case studies first")
    return json.loads(path.read_text())


def test_source_motion_is_original_state_reconstruction_at_original_rate(manifest):
    total=0
    for case in manifest["cases"]:
        source=case["source_replay"]
        assert source["role"]=="original_source_robot_A_motion"
        assert source["state_reconstruction_used"] and not source["scripted_controller_used"] and not source["action_replay_used"]
        assert source["control_hz"]==source["video_fps"]==20
        assert source["speed_multiplier"]==1
        assert source["validated_frames"]==source["encoded_frames"]==source["state_shape"][0]
        assert source["max_eef_position_error_m"]<1e-8 and source["max_can_position_error_m"]<1e-8
        path=OUT/"source"/f"source_{case['source_demo_id']}_frame_validation.csv"
        rows=list(csv.DictReader(path.open()))
        assert len(rows)==source["validated_frames"]
        assert max(float(row["eef_error_m"]) for row in rows)<1e-8
        assert max(float(row["can_error_m"]) for row in rows)<1e-8
        total+=len(rows)
    assert total==347


def test_training_lineage_uses_one_shared_model_and_all_ten_target_teachers(manifest):
    rows=list(csv.DictReader((OUT/"training_lineage.csv").open()))
    assert len(rows)==manifest["training_teacher_episodes"]==10
    assert sum(int(row["training_samples"]) for row in rows)==manifest["training_state_action_samples"]==1620
    assert {row["source_demo_id"] for row in rows}==set(manifest["training_source_demo_ids"])
    assert len(manifest["training_source_demo_ids"])==5
    hashes={row["shared_checkpoint_sha256"] for row in rows}
    assert hashes=={manifest["shared_checkpoint_sha256"]}
    train_seeds={int(row["target_teacher_seed"]) for row in rows}
    for case in manifest["cases"]:
        assert case["learned_policy_is_shared_across_all_cases"]
        assert case["shared_checkpoint_sha256"] in hashes
        assert case["learned_B"]["seed"] not in train_seeds
        contributions=case["source_contribution_to_shared_training"]
        assert len(contributions)==2 and all(row["source_demo_id"]==case["source_demo_id"] for row in contributions)
        assert case["training_teacher"]["seed"] in {int(row["target_teacher_seed"]) for row in contributions}


def test_english_cases_preserve_original_outcomes_and_decode_verified_videos(manifest):
    assert len(manifest["cases"])==len(manifest["movies"])==3
    for case in manifest["cases"]:
        for replay in [case["training_teacher"],case["learned_B"]]:
            assert replay["video_language"]=="en" and replay["same_initial_features"]
            assert replay["source_original_success"] and replay["terminal"]["native_success"]
            assert replay["original_terminal_position_delta_m"]<1e-8
            assert replay["stable_placement_1s"] and replay["post_hold_seconds"]==1
        assert (OUT/case["case_video"]).is_file()
    for item in manifest["verified_media"]:
        path=OUT/item["path"]
        assert path.stat().st_size==item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
        assert item["frames"]>0 and item["duration_seconds"]>0
    for movie in manifest["movies"]:
        assert movie["width"]==1920 and movie["height"]==1080 and movie["fps"]==20
        assert movie["overlay_language"]=="English"
        transcript=OUT/"videos"/Path(movie["overlay_transcript"]).name
        text=transcript.read_text()
        assert text.isascii() and "ONE shared checkpoint" in text


def test_english_report_pdf_matches_successful_layout_and_font_audit():
    folder=ROOT/"reports/english_technical_report_20261106"
    if not (folder/"qa/qa_report.json").exists():pytest.skip("Build English technical PDF first")
    qa=json.loads((folder/"qa/qa_report.json").read_text())
    pdf=folder/"Cross_Robot_Real2Sim_Technical_Report.pdf"
    assert qa["status"]=="passed" and not qa["issues"]
    assert qa["english_text_check"]=="passed" and qa["page_count"]>=15
    assert all(font["embedded"] for font in qa["embedded_fonts"])
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()==qa["pdf_sha256"]
