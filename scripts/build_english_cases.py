#!/usr/bin/env python3
"""English source-A / training-demonstration / learned-B case studies."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN_ID="fast_20261105T182127Z"
CASE_SPECS=[("01","demo_3",1042,10001),("02","demo_5",1043,10003),("03","demo_7",1044,10009)]


def write(path,payload):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(payload,indent=2)+"\n");temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job-id",default="english_cases_20261106")
    parser.add_argument("--phase",choices=("replays","all"),default="all")
    args=parser.parse_args()
    if not args.job_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.job_id):
        parser.error("Invalid job ID")
    raw=ROOT/"runs"/args.job_id
    output=ROOT/"results/english_case_studies_20261106"
    raw.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    status={"status":"running","phase":args.phase,"job_id":args.job_id,"pid":os.getpid(),
            "started_at":dt.datetime.now(dt.timezone.utc).isoformat(),"exit_code":None}
    lock=(ROOT/"runs/.gpu0.lock").open("a")
    try:
        status["status"]="waiting_for_project_gpu_lock";write(raw/"status.json",status)
        fcntl.flock(lock,fcntl.LOCK_EX)
        status["status"]="running";write(raw/"status.json",status)
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
        import torch
        from cross_robot_real2sim.common.provenance import file_sha256
        from cross_robot_real2sim.viz.review_rollouts import replay_case
        torch.set_num_threads(2)
        run=ROOT/"runs"/RUN_ID
        checkpoint=run/"04_policy/checkpoint.pt"
        model=torch.load(checkpoint,map_location="cpu",weights_only=False)
        metadata=model["dataset_metadata"]
        lineage=[]
        for ep in metadata["episodes"]:
            lineage.append({"source_demo_id":ep["source_demo_id"],"target_teacher_episode_id":ep["episode_id"],
                            "target_teacher_seed":ep["seed"],"training_samples":ep["samples"],
                            "teacher_origin":ep["teacher_origin"],"success":ep["success"],
                            "trajectory_npz":str(Path(ep["path"]).relative_to(ROOT)),
                            "trajectory_sha256":file_sha256(ep["path"]),"shared_checkpoint_sha256":file_sha256(checkpoint)})
        with (output/"training_lineage.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(lineage[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(lineage)
        cases=[]
        for number,demo_id,teacher_seed,eval_seed in CASE_SPECS:
            status["current_case"]=number;write(raw/"status.json",status)
            teacher_request={"case_id":f"teacher_{demo_id}_seed{teacher_seed}","run_id":RUN_ID,"seed":teacher_seed,
                             "teacher_origin":"source_retargeted","source_demo_id":demo_id,"language":"en",
                             "original_npz":f"runs/{RUN_ID}/03_retarget/rollouts/UR5e_source_retargeted_{teacher_seed}.npz",
                             "variant_label":f"training demonstration from {demo_id}"}
            policy_request={"case_id":f"learned_shared_policy_seed{eval_seed}","run_id":RUN_ID,"seed":eval_seed,
                            "teacher_origin":"learned_policy","language":"en",
                            "checkpoint":f"runs/{RUN_ID}/04_policy/checkpoint.pt",
                            "original_npz":f"runs/{RUN_ID}/04_policy/eval_rollouts/eval_{eval_seed}.npz",
                            "variant_label":"shared BC model; held-out evaluation"}
            teacher=replay_case(ROOT,output,raw,teacher_request)
            learned=replay_case(ROOT,output,raw,policy_request)
            if not all(x["terminal"]["native_success"] and x["stable_placement_1s"] for x in (teacher,learned)):
                raise ValueError(f"Case {number} does not meet success and same-policy stability gates")
            source_meta_path=output/"source"/f"source_{demo_id}.json"
            if not source_meta_path.exists():raise ValueError(f"Authentic source reconstruction is missing: {source_meta_path}")
            source=json.loads(source_meta_path.read_text())
            cases.append({"case_id":number,"source_demo_id":demo_id,"source_replay":source,
                          "source_video":f"source/source_{demo_id}.mp4",
                          "training_teacher":teacher,"learned_B":learned,
                          "source_contribution_to_shared_training":[row for row in lineage if row["source_demo_id"]==demo_id],
                          "shared_checkpoint":str(checkpoint.relative_to(ROOT)),"shared_checkpoint_sha256":file_sha256(checkpoint),
                          "learned_policy_is_shared_across_all_cases":True,
                          "case_video":f"videos/case_{number}_{demo_id}_to_ur5e.mp4"})
        manifest={"schema_version":"1.0","created_at":dt.datetime.now(dt.timezone.utc).isoformat(),
                  "source_run_id":RUN_ID,"video_language":"English","human_video_used":False,"new_training_performed":False,
                  "observation_mode":"oracle_sim_state","source_dataset":"robomimic/robomimic_datasets/v1.5/can/ph/low_dim_v15.hdf5",
                  "training_source_demo_ids":sorted({row["source_demo_id"] for row in lineage}),
                  "training_teacher_episodes":len(lineage),"training_state_action_samples":metadata["samples"],
                  "shared_checkpoint_sha256":file_sha256(checkpoint),"model_config":model["model_config"],"training_steps":model["step"],
                  "case_relationship":"Three source-lineage demonstrations in ONE shared policy; not three separately trained models or a one-source-only causal comparison.",
                  "cases":cases}
        write(output/"case_manifest.json",manifest)
        if args.phase=="all":
            from cross_robot_real2sim.viz.english_case_studies import assemble_cases
            status["current_case"]="assembling_english_case_videos";write(raw/"status.json",status)
            assemble_cases(ROOT,output,manifest)
        status.update(status="completed",exit_code=0,finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        write(raw/"status.json",status)
        return 0
    except BaseException as exc:
        status.update(status="failed",exit_code=1,exception=repr(exc),traceback=traceback.format_exc(),finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        write(raw/"status.json",status);traceback.print_exc();return 1
    finally:lock.close()


if __name__=="__main__":raise SystemExit(main())
