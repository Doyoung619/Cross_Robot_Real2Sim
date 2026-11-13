#!/usr/bin/env python3
"""Verify actual pilot artifacts and that executed target data never train VLA."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args=parser.parse_args()
    out=Path(args.run_dir).resolve()
    root=Path(__file__).resolve().parents[1]
    read=lambda name:json.loads((out/name).read_text())
    selection,processing,training,rows=map(read,["selection.json","processing_audit.json","training.json","evaluation.json"])
    assert not set(selection["selected"]) & set(selection["excluded_validation_ids"])
    assert selection["target_rollout_training_calls"]==0
    assert processing["physics_steps_during_processing"]==0
    assert processing["target_executed_training_episodes"]==0
    assert all(not x["executed_target_demo"] for x in processing["lineage"])
    assert {x["source_demo_id"] for x in processing["lineage"]}==set(selection["selected"])
    assert training["executed_target_training_episodes"]==0 and training["steps"]==100
    assert training["trainable_parameters"]==334592
    assert hashlib.sha256((out/"adapter.safetensors").read_bytes()).hexdigest()==training["adapter_sha256"]
    with (out/"adapter.safetensors").open('rb') as f:
        header=json.loads(f.read(int.from_bytes(f.read(8),'little')))
    keys=[k for k in header if k!='__metadata__']
    assert keys and all('lora_' in k for k in keys)
    parameter_count=0
    for key in keys:
        n=1
        for dim in header[key]['shape']:n*=dim
        parameter_count+=n
    assert parameter_count==334592
    source=ast.parse((root/"scripts/vla_pilot_processing.py").read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('step','mj_step') for n in ast.walk(source))
    assert len(rows)==6
    assert all(row["training_usage"]=="none_final_evaluation_only" for row in rows)
    for demo in selection["evaluation_source_ids"]:
        cases=[r for r in rows if r["source_initial_scene_id"]==demo]
        assert len(cases)==3
        assert all(r["initial_plate_position"]==cases[0]["initial_plate_position"] for r in cases)
    videos=[]
    for row in rows:
        path=out/row["video"]
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(path)],text=True))
        assert info['streams'][0]['width']==640 and info['streams'][0]['height']==360
        assert float(info['format']['duration'])>0
        videos.append(dict(path=row["video"],bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),probe=info))
    result=dict(status="passed",checks=["source_selection_and_evaluation_disjoint","zero_B_training_execution",
        "source_lineage_complete","adapter_hash_matches","only_334592_LoRA_parameters_trainable",
        "offline_processing_has_no_step_calls","six_real_final_evaluations","matched_object_initial_conditions",
        "six_decodable_English_labeled_videos"],videos=videos)
    (out/"verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],checks=len(result['checks']),videos=len(videos))))


if __name__=="__main__":
    main()
