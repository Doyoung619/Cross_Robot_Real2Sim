"""Guard feedback reporting against mixing development and reserved-test scores."""
import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "package_vla_feedback", Path(__file__).resolve().parents[1] / "scripts/package_vla_feedback.py")
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def write_rows(root, directory, rows):
    path = root / directory
    path.mkdir(parents=True)
    (path / "evaluation.json").write_text(json.dumps(rows))


def test_final_policies_and_control_are_never_averaged_into_feedback_rounds(tmp_path):
    write_rows(tmp_path, "rounds/baseline", [dict(condition="baseline", success=False, goal_error_m=.2)])
    write_rows(tmp_path, "rounds/round_1", [dict(condition="round1", success=True, goal_error_m=.01)])
    write_rows(tmp_path, "final", [dict(condition="baseline", success=False, goal_error_m=.3),
                                    dict(condition="selected", success=True, goal_error_m=.02)])
    write_rows(tmp_path, "control", [dict(condition="no_feedback", success=False, goal_error_m=.15)])
    entries = PACKAGE.discover(tmp_path)
    assert len(entries) == 5
    final = [e for e in entries if e["role"] == "final"]
    assert len(final) == 2
    assert {PACKAGE.mean(e["rows"], "goal_error_m") for e in final} == {.3, .02}
    assert sum(e["role"] == "control" for e in entries) == 1
    assert [e["name"] for e in entries if e["role"] == "development"] == ["baseline", "round_1"]


def test_missing_contact_measurement_is_not_reported_as_zero(tmp_path):
    write_rows(tmp_path, "rounds/baseline", [dict(condition="baseline", success=False, goal_error_m=.2)])
    table = "\n".join(PACKAGE.measured_table(PACKAGE.discover(tmp_path)))
    assert "not recorded" in table
    assert "0.00%" not in table
    assert "20.00 cm" in table


def test_report_does_not_claim_no_target_supervision_without_provenance(tmp_path):
    raw = tmp_path / "raw"
    write_rows(raw, "rounds/round_1", [dict(condition="round1", success=False, goal_error_m=.2)])
    dest = tmp_path / "result"
    dest.mkdir()
    PACKAGE.report_markdown(raw, dest, PACKAGE.discover(raw), [])
    text = (dest / "FINAL_REPORT.md").read_text()
    assert "No executed B trajectory is used as action supervision" not in text
    assert "not independent generalization evidence" in text
    assert "No matched no-feedback control is available" in text
