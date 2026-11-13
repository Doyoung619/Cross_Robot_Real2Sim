#!/usr/bin/env python3
"""Ten-minute source-only adapter continuation and fixed before/after eval."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

from vla_pilot_common import ROOT, bootstrap, write_json


def package(out,parent,rows):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    target=ROOT/"results"/out.name
    (target/"videos").mkdir(parents=True,exist_ok=True)
    training=json.loads((out/"training.json").read_text())
    selection=json.loads((out/"selection.json").read_text())
    previous=[r for r in rows if r["condition"]=="B_previous100_UR5e"]
    extended=[r for r in rows if r["condition"]=="B_extra10min_UR5e"]
    mean=lambda subset:sum(r["goal_error_m"] for r in subset)/len(subset)
    comparison=dict(previous_successes=sum(r["success"] for r in previous),
                    extended_successes=sum(r["success"] for r in extended),episodes_each=len(previous),
                    previous_mean_goal_error_m=mean(previous),extended_mean_goal_error_m=mean(extended),
                    same_source_initial_scenes=selection["evaluation_source_ids"],
                    target_training_demonstrations=0,selection_and_processing_unchanged=True,
                    parent_adapter_preserved=True,
                    target_test_set_reused=True,statistical_generalization_claimed=False)
    write_json(out/"comparison.json",comparison)
    for name in ["training.json","train_log.json","evaluation.json","comparison.json","request.json","selection.json","adapter.safetensors"]:
        shutil.copyfile(out/name,target/name)
    for row in rows:shutil.copyfile(out/row["video"],target/row["video"])
    for demo in selection["evaluation_source_ids"]:
        pair=[next(r for r in rows if r["condition"]==name and r["source_initial_scene_id"]==demo)
              for name in ["B_previous100_UR5e","B_extra10min_UR5e"]]
        readers=[imageio.get_reader(out/r["video"]) for r in pair]
        lengths=[r.count_frames() for r in readers]
        fontpath="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        title=ImageFont.truetype(fontpath,29);body=ImageFont.truetype(fontpath,22)
        movie=target/"videos"/f"training_duration_comparison_{demo}.mp4"
        try:
            with imageio.get_writer(str(movie),fps=10,codec="libx264",quality=8,macro_block_size=1) as writer:
                for i in range(max(lengths)):
                    image=Image.new("RGB",(1280,800),(12,20,32));d=ImageDraw.Draw(image)
                    d.text((25,22),"Same source data | 100 updates vs 10 more minutes",font=title,fill="white")
                    d.text((25,70),f"3 A demos, 189 processed anchors, 0 executed B training demos | {demo}",font=body,fill=(180,213,240))
                    for j,(reader,row) in enumerate(zip(readers,pair)):
                        image.paste(Image.fromarray(reader.get_data(min(i,lengths[j]-1))),(640*j,150))
                        d.text((640*j+25,550),f"Success: {row['success']} | error: {row['goal_error_m']*100:.2f} cm",font=body,fill="white")
                    d.text((25,635),f"Actual extra training: {training['additional_training_seconds']:.1f}s; updates: {training['additional_steps']}",font=body,fill="white")
                    d.text((25,680),"Same fixed test starts and inference settings; no target trajectories used for training.",font=body,fill=(180,213,240))
                    d.text((25,725),"Repeated test scenes: this is a duration comparison, not a new generalization benchmark.",font=body,fill=(255,210,150))
                    writer.append_data(np.asarray(image))
        finally:
            for r in readers:r.close()
    text=("# Ten-minute B adapter continuation\n\n"
          "Task: push the plate to the front of the stove. The prior 100-update adapter weights are the starting point.\n\n"
          f"Extra training: **{training['additional_training_seconds']:.2f} seconds**, **{training['additional_steps']} new updates**, **{training['steps']} total updates**.\n\n"
          "The same 189 source-derived static-rendering anchors from three A episodes are reused. No executed B training demonstrations or new data are added. LoRA targets, learning rate, batch size, cameras, normalizers and inference settings remain fixed.\n\n"
          "| B policy | Successes / trials | Mean final XY goal error |\n|---|---:|---:|\n"
          f"| Previous 100 updates (re-evaluated) | {comparison['previous_successes']}/{len(previous)} | {mean(previous)*100:.2f} cm |\n"
          f"| Plus ten minutes | {comparison['extended_successes']}/{len(extended)} | {mean(extended)*100:.2f} cm |\n\n"
          "## English videos\n\n"+
          "\n".join(f"- [{demo}: previous vs extended training](videos/training_duration_comparison_{demo}.mp4)" for demo in selection["evaluation_source_ids"])+
          "\n\n## Interpretation limits\n\n"
          "The initial pilot saved adapter weights but no optimizer state. This continuation therefore starts from those exact weights with a fresh AdamW optimizer; it is not an exact optimizer/RNG resume. New checkpoints include optimizer and RNG state. Uniform sampling is with replacement, so update counts and equivalent sample exposures are reported instead of exact epochs.\n\n"
          "The checkpoint is the last one at the fixed time budget, never the best target-score checkpoint. Target results are evaluation only. These same two test scenes were already examined; the result cannot establish broad generalization or isolate optimizer-state effects. More updates also cannot validate contact dynamics missing from the synthetic source processing.\n\n"
          f"Parent adapter SHA256: `{training['parent_adapter_sha256']}`.\n\n"
          f"New adapter SHA256: `{training['adapter_sha256']}`.\n\n"
          f"Processed training data SHA256: `{training['processed_training_data_sha256']}`.\n")
    (out/"FINAL_REPORT.md").write_text(text);(target/"FINAL_REPORT.md").write_text(text)
    (target/"README.md").write_text(text)
    write_json(target/"artifact_manifest.json",dict(files=[dict(path=str(p.relative_to(target)),bytes=p.stat().st_size,
               sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(target.rglob("*")) if p.is_file() and p.name!="artifact_manifest.json"]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run-dir",required=True);parser.add_argument("--deadline",required=True)
    args=parser.parse_args();out=Path(args.run_dir).resolve()
    request=json.loads((out/"request.json").read_text());parent=Path(request["parent_run"])
    deadline=datetime.fromisoformat(args.deadline.replace("Z","+00:00")).timestamp()
    ledger=dict(events=[],parent_run=str(parent),request=request)
    def progress(stage,details):
        event=dict(stage=stage,at=datetime.now(timezone.utc).isoformat(),details=details)
        ledger["events"].append(event);ledger["current_stage"]=stage;write_json(out/"progress.json",ledger)
        print(json.dumps(event,default=str),flush=True)
    try:
        bootstrap()
        from vla_pilot_learning import continue_for_duration
        from vla_pilot_evaluate import evaluate
        for name in ["selection.json","camera_calibration.npz"]:shutil.copyfile(parent/name,out/name)
        progress("initializing",dict(parent_adapter=str(parent/"adapter.safetensors")))
        continue_for_duration(out,parent,progress,seconds=request["additional_training_seconds"])
        rows=evaluate(out,progress,deadline,horizon=200,conditions=[
            ("B_previous100_UR5e","UR5e",True,parent/"adapter.safetensors"),
            ("B_extra10min_UR5e","UR5e",True,out/"adapter.safetensors")])
        package(out,parent,rows)
        progress("completed",dict(report=str(out/"FINAL_REPORT.md")))
    except Exception:
        error=traceback.format_exc();progress("failed",dict(traceback=error))
        (out/"FINAL_REPORT.md").write_text("# Ten-minute continuation failed\n\n```text\n"+error+"\n```\n")
        raise


if __name__=="__main__":main()
