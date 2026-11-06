#!/usr/bin/env python3
"""Detached-job entry point for verified original Robot A source videos."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", nargs="+", default=["demo_3", "demo_5", "demo_7"])
    parser.add_argument("--dataset", default=str(REPO/"data/source/v1.5/can/ph/low_dim_v15.hdf5"))
    parser.add_argument("--output-dir", default=str(REPO/"results/english_case_studies_20261106/source"))
    parser.add_argument("--scratch-dir", default=str(REPO/"runs/english_source_replay_20261106"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    scratch = Path(args.scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
    status_path = scratch/"status.json"
    status = {"status": "waiting_for_gpu_lock", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
              "demos": args.demos, "results": []}
    def persist():
        status_path.write_text(json.dumps(status, indent=2))
    persist()
    print("Waiting for project GPU 0 lock", flush=True)
    lock_path=REPO/"runs/.gpu0.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status["status"]="running"; persist()
            print("Acquired project GPU 0 lock", flush=True)
            from cross_robot_real2sim.viz.source_dataset_replay import render_source_demo
            for demo in args.demos:
                print(f"Rendering ORIGINAL recorded Panda {demo}", flush=True)
                result=render_source_demo(args.dataset, demo, Path(args.output_dir)/f"source_{demo}.mp4",
                                          {"scratch_dir":str(scratch),"width":640,"height":480})
                status["results"].append(result);persist()
                print(f"Verified {demo}: {result['encoded_frames']} frames, EEF error {result['max_eef_position_error_m']:.3g} m, Can error {result['max_can_position_error_m']:.3g} m",flush=True)
            status["status"]="completed";status["exit_code"]=0;persist()
            print("All original source videos completed; releasing GPU 0 lock", flush=True)
    except BaseException as exc:
        status.update(status="failed", exit_code=1,error=str(exc),traceback=traceback.format_exc())
        persist();raise
    finally:
        status["finished_at"]=datetime.now(timezone.utc).isoformat();persist()


if __name__=="__main__": main()
