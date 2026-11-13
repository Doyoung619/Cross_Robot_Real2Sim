#!/usr/bin/env python3
"""Bounded, source-only A-VLA to B-VLA adaptation pilot with real evaluation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

from vla_pilot_common import bootstrap,write_json,ROOT,TASK


def report(out,ledger,error=None):
    rows=json.loads((out/"evaluation.json").read_text()) if (out/"evaluation.json").exists() else []
    selection=json.loads((out/"selection.json").read_text()) if (out/"selection.json").exists() else None
    training=json.loads((out/"training.json").read_text()) if (out/"training.json").exists() else None
    lines=["# One-hour source-only VLA transfer pilot","",f"Task: **{TASK}**.","",
           "A: Panda (7 arm joints). B: UR5e (6 arm joints), Robotiq85 gripper.","",
           "Training uses only existing source records transformed by offline B kinematics and static rendering. **No executed B training demonstrations.**","",
           "This is a minimal feasibility experiment, not evidence of broad generalization.","",
           "## Measured final evaluation","",
           "| Condition | Successful / evaluated | Mean final XY goal error |",
           "|---|---:|---:|"]
    for name in ["A_pretrained_Panda","B_pretrained_UR5e","B_adapted_UR5e"]:
        subset=[r for r in rows if r["condition"]==name]
        mean=sum(r["goal_error_m"] for r in subset)/len(subset) if subset else None
        lines.append(f"| {name} | {sum(r['success'] for r in subset)}/{len(subset)} | {mean:.4f} m |" if mean is not None else f"| {name} | not measured | not measured |")
    lines += ["","A source-policy success is a prerequisite to claiming transfer. Positive human-demo terminal rewards do not prove VLA source competence.",""]
    a=[r for r in rows if r["condition"]=="A_pretrained_Panda"]
    b=[r for r in rows if r["condition"]=="B_pretrained_UR5e"]
    c=[r for r in rows if r["condition"]=="B_adapted_UR5e"]
    if a and not any(r["success"] for r in a):
        lines += ["**Source-policy gate failed:** the downloaded A checkpoint did not complete this task under the tested integration. This run does not establish transfer from an already competent A policy.",""]
    if b and c:
        improvement=sum(r["success"] for r in c)/len(c)-sum(r["success"] for r in b)/len(b)
        lines += [f"Measured adapted-minus-unadapted B success-rate difference: **{improvement:+.3f}**. With only {len(c)} trials, this is descriptive, not a reliable performance estimate.",""]
    lines += ["## Data and model provenance","",f"Source selection: `{selection}`.","",f"Adapter training: `{training}`.","",
              "The frozen pretrained checkpoint is lerobot/smolvla_libero revision a24f1f380461b665158d52acd140152c79e8b36d. Source data is original LIBERO plate-pushing demonstrations.","",
              "Static rendering copies source object states and solves hypothetical B poses. It does not establish physically correct B contact dynamics. Synthetic images are labeled as source-derived, not B execution recordings.","",
              "Both B conditions use the same fixed source-frame camera/state calibration. The pretrained-B baseline has no learned weight adaptation. Evaluation outputs are never read by training.","",
              "Control: world-frame relative OSC_POSE, 20 Hz, 7D Cartesian+gripper action. Three source episodes maximum, 100 LoRA updates, batch four, rank eight. Source state records provide the geometry; synthetic training and live evaluation render 256px images with the official 180-degree image rotation, followed by the saved 512px VLA preprocessing. No human video reconstruction.","",
              "No independent hyperparameter search or source-validation checkpoint selection in this minimal run. An unselected-source control is also absent, so selection benefit cannot be isolated.","",
              "## English videos",""]
    for row in rows:
        lines.append(f"- [{row['condition']} / {row['source_initial_scene_id']} / success={row['success']}]({row['video']})")
    if error:
        lines += ["","## Execution failure","",f"```text\n{error}\n```",""]
    lines += ["","## Audit artifacts","","- `selection_candidates.json`, `selection.json`: offline B-aware selection.",
              "- `processing_audit.json`: per-frame IK checks and source lineage; zero target physics steps.",
              "- `training.json`, `train_log.json`, `adapter.safetensors`: source-initialized LoRA update evidence.",
              "- `evaluation.json`, `*_actions.json`: final closed-loop results, never training inputs.",
              "- `worker.log`, `progress.json`, `status.json`: execution and time-budget evidence.",""]
    (out/"FINAL_REPORT.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--deadline",required=True)
    args=parser.parse_args()
    out=Path(args.run_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(args.deadline.replace("Z","+00:00")).timestamp()
    ledger=dict(started_at=datetime.now(timezone.utc).isoformat(),task=TASK,deadline=args.deadline,
                protocol="source_only_no_executed_B_demonstrations",events=[])
    def progress(stage,details):
        event=dict(stage=stage,at=datetime.now(timezone.utc).isoformat(),details=details)
        ledger["events"].append(event)
        ledger["current_stage"]=stage
        write_json(out/"progress.json",ledger)
        print(json.dumps(event,default=str),flush=True)
    try:
        bootstrap()
        from vla_pilot_processing import process
        from vla_pilot_learning import train
        from vla_pilot_evaluate import evaluate
        progress("started",dict(time_remaining_seconds=int(deadline-time.time())))
        if not (out/"processed_source_data.npz").exists():
            process(out,progress)
        if not (out/"adapter.safetensors").exists():
            train(out,progress,steps=100)
        rows=evaluate(out,progress,deadline,horizon=200)
        progress("completed",dict(evaluation_episodes=len(rows)))
        report(out,ledger)
    except Exception:
        error=traceback.format_exc()
        progress("failed",dict(traceback=error))
        report(out,ledger,error)
        raise


if __name__=="__main__":
    main()
