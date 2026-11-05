#!/usr/bin/env python3
"""Create reproducible review media from frozen runs; no training is performed."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
FAST="fast_20261105T182127Z"
BOOT="bootstrap_20261105T181834Z"


def case_requests():
    def policy(run_id,seed,after=False):
        folder="05_diagnosis/feedback_run/04_policy" if after else "04_policy"
        label="피드백 후" if after else "피드백 전"
        return {"case_id":f"{run_id.split('_')[0]}_bc_{'after' if after else 'before'}_{seed}",
                "run_id":run_id,"seed":seed,"teacher_origin":"learned_policy",
                "checkpoint":f"runs/{run_id}/{folder}/checkpoint.pt",
                "config_path":f"{folder}/effective_config.yaml",
                "original_npz":f"runs/{run_id}/{folder}/eval_rollouts/eval_{seed}.npz",
                "variant_label":label}
    stats=json.loads((ROOT/"runs"/FAST/"03_retarget/stats.json").read_text())
    teacher=next(e for e in stats["episodes"] if e["teacher_origin"]=="source_retargeted")
    panda=next(e for e in stats["episodes"] if e["teacher_origin"]=="source_scripted")
    requests=[{"case_id":"fast_panda_scripted_42","run_id":FAST,"seed":42,"robot":"Panda",
               "teacher_origin":"source_scripted","original_npz":str(Path(panda["path"]).relative_to(ROOT)),
               "variant_label":"별도로 실행한 규칙 기반 Panda; HDF5 재생 영상 아님"},
              {"case_id":"fast_teacher_1042","run_id":FAST,"seed":teacher["seed"],"teacher_origin":"source_retargeted",
               "source_demo_id":teacher["source_demo_id"],"original_npz":str(Path(teacher["path"]).relative_to(ROOT)),
               "variant_label":"Panda 데모의 잡기 위치·들기 높이 사용"},
              {"case_id":"fast_teacher_10000","run_id":FAST,"seed":10000,"teacher_origin":"source_retargeted",
               "source_demo_id":teacher["source_demo_id"],"variant_label":"학습 정책 실패 시드의 교사 진단; 학습에 사용하지 않음"}]
    requests.extend(policy(FAST,seed) for seed in range(10000,10010))
    requests.extend(policy(FAST,seed,after=True) for seed in (10000,10001))
    boot_stats=json.loads((ROOT/"runs"/BOOT/"03_retarget/stats.json").read_text())
    teacher=next(e for e in boot_stats["episodes"] if e["teacher_origin"]=="source_retargeted")
    requests.append({"case_id":"bootstrap_teacher_1042","run_id":BOOT,"seed":teacher["seed"],"teacher_origin":"source_retargeted",
                     "source_demo_id":teacher["source_demo_id"],"original_npz":str(Path(teacher["path"]).relative_to(ROOT)),
                     "variant_label":"bootstrap 원본 데모 기반 교사"})
    requests.extend([policy(BOOT,10002),policy(BOOT,10002,after=True)])
    return requests


def write(path,payload):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job-id",required=True)
    parser.add_argument("--output",default="results/review_20261105")
    parser.add_argument("--phase",choices=("replays","all"),default="all")
    args=parser.parse_args()
    if not args.job_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.job_id):
        parser.error("Invalid job ID")
    raw=ROOT/"runs"/args.job_id
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/"results"):
        parser.error("Review output must be inside this repository's results directory")
    raw.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    state={"job_id":args.job_id,"status":"running","phase":args.phase,"pid":os.getpid(),
           "started_at":dt.datetime.now(dt.timezone.utc).isoformat(),"output":str(output),"exit_code":None}
    status_path=raw/"status.json"
    write(status_path,state)
    lock=(ROOT/"runs/.gpu0.lock").open("a")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
        import torch
        torch.set_num_threads(2)
        def stop(signum,frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM,stop)
        from cross_robot_real2sim.viz.review_rollouts import replay_case
        requests=case_requests()
        results=[]
        for index,request in enumerate(requests):
            state.update(current_case=request["case_id"],completed_cases=index,total_cases=len(requests))
            write(status_path,state)
            results.append(replay_case(ROOT,output,raw,request))
        columns=["case_id","source_run_id","seed","teacher_origin","source_original_success","replay_success",
                 "stable_placement_1s","terminal_xy_error_m","post_hold_xy_error_m","same_initial_features","video_path"]
        with (output/"cases.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n");writer.writeheader()
            for item in results:
                row={key:item.get(key) for key in columns}
                row.update(replay_success=item["terminal"]["native_success"],terminal_xy_error_m=item["terminal"]["xy_error_m"],
                           post_hold_xy_error_m=item["post_hold"]["xy_error_m"])
                writer.writerow(row)
        write(output/"replay_manifest.json",{"schema_version":"1.0","source_runs":[FAST,BOOT],"no_new_training":True,
              "human_video_used":False,"observation_mode":"oracle_sim_state","replays":results,
              "note":"Frozen checkpoints/controllers re-executed. First two stage cards use actual saved metadata. Replays are excluded from training."})
        if args.phase=="all":
            from cross_robot_real2sim.viz.review_story import assemble
            state.update(current_case="assembling_stage_videos");write(status_path,state)
            assemble(ROOT,output,raw,results)
        state.update(status="completed",completed_cases=len(results),finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),exit_code=0)
        write(status_path,state)
        return 0
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",exception=repr(exc),
                     traceback=traceback.format_exc(),finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),exit_code=1)
        write(status_path,state)
        traceback.print_exc()
        return 1
    finally:
        lock.close()


if __name__=="__main__":
    raise SystemExit(main())
