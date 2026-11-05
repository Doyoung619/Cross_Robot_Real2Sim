"""Executed Panda source policy and source-grounded UR5e synthetic collection."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import numpy as np
from ..adapters.source_hdf5 import ensure_dataset, list_demonstrations
from .source_trace import load_source_trace
from .rollouts import collect_episode, save_episode


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x)))


def run(config, run_dir):
    started = time.monotonic()
    run_dir = Path(run_dir)
    out = run_dir / "03_retarget"
    out.mkdir(parents=True,exist_ok=True)
    # A restarted incomplete stage is reconstructed, while all previous executed
    # evidence remains in history. Current policy training sees only this pass.
    if (out/"rollouts").exists() and any((out/"rollouts").glob("*.npz")):
        import shutil
        archive=out/"history"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive.mkdir(parents=True)
        for name in ("rollouts","attempts.csv","episodes.json","stats.json","status.json","exceptions.json","source_trace.npz"):
            if (out/name).exists(): shutil.move(str(out/name),str(archive/name))
    (out / "rollouts").mkdir(exist_ok=True)
    videos = run_dir / "videos"; videos.mkdir(parents=True, exist_ok=True)
    status = {"schema_version":"1.0", "seed":int(config.get("seed",42)),
              "mode":config.get("mode","bootstrap"), "status":"running",
              "started_at":datetime.now(timezone.utc).isoformat(), "privileged_state_used":True,
              "observation_mode":"oracle_sim_state", "controller":"BASIC/OSC_POSE"}
    _write(out/"status.json",status)
    selection_path = run_dir/"01_selection/selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    task_path = run_dir/"02_task/task_spec.json"
    if not task_path.exists() and (run_dir.parent.parent/"02_task/task_spec.json").exists():
        task_path = run_dir.parent.parent/"02_task/task_spec.json"
    if not task_path.exists():
        # Feedback runs may reuse the original immutable task by explicit path.
        task_path = Path(config.get("task_spec_path",task_path))
    from ..schemas import TaskSpec
    task = TaskSpec.model_validate_json(task_path.read_text()).model_dump()
    # This MVP implements only Can-inside-bin. Refuse to silently run a different
    # task if real-video interpretation or the manual task seed changes its goal.
    object_label = task.get("object", "can")
    if isinstance(object_label,dict): object_label = object_label.get("label",object_label.get("name","can"))
    if str(object_label).lower() != "can":
        raise ValueError(f"Unsupported task object from task JSON: {object_label}")
    if task["goal"] != {"relation":"inside","receptacle":"bin"}:
        raise ValueError(f"Unsupported parsed goal: {task['goal']}")
    expected_map={"can":"Can", "bin":"PickPlaceCan.target_bin_placements[can_index]"}
    for label,simulator_id in expected_map.items():
        if task["semantic_to_simulator"].get(label) != simulator_id:
            raise ValueError(f"Unsupported semantic mapping {label} -> {task['semantic_to_simulator'].get(label)}")
    dataset = Path(selection.get("source_dataset_path") or ensure_dataset(config))
    ids = selection.get("selected_source_demonstration_ids") or config.get("selected_source_demonstration_ids") or [d["demo_id"] for d in list_demonstrations(dataset,config.get("selected_source_demo_limit",2))]
    traces, errors = [], []
    for demo_id in ids:
        try:
            trace = load_source_trace(dataset,demo_id,config.get("control_hz",20))
            trace["metadata"].update({"seed":status["seed"],"mode":status["mode"],"input_task_path":str(task_path),"selection_path":str(selection_path)})
            traces.append(trace)
            if len(traces)==1:
                np.savez_compressed(out/"source_trace.npz",**{k:v for k,v in trace.items() if k!="metadata"},metadata_json=np.array(json.dumps(trace["metadata"])))
            _write(out/f"source_trace_{demo_id}.json",trace["metadata"])
        except Exception as exc:
            errors.append({"component":"source_trace", "demo_id":demo_id,"error":str(exc),"traceback":traceback.format_exc()})
    episodes=[]
    attempts=[]
    def execute(seed,origin,robot="UR5e",trace=None,video=None):
        try:
            result=collect_episode(config,seed,origin,source_trace=trace,video_path=video,robot=robot)
            episode=save_episode(result,out/"rollouts"/f"{robot}_{origin}_{seed}.npz")
            episodes.append(episode)
            row={k:episode.get(k) for k in ("episode_id","seed","robot","teacher_origin","source_demo_id","success","reason_code","placement_error_m","max_lift_m","had_grasp_contact","steps","final_phase","duration_seconds","path")}
            row.update({"schema_version":"1.0","mode":status["mode"],"privileged_state_used":True,"diagnostic_oracle_used":True,"controller":"BASIC/OSC_POSE"})
            attempts.append(row)
            print(f"[stage3] {robot} {origin} seed={seed} success={episode['success']} error={episode['placement_error_m']:.4f}m reason={episode['reason_code']}",flush=True)
            # Persist after every complete episode so interruption loses at most
            # the current trial and never erases failed-trial evidence.
            _write(out/"episodes.json",episodes)
            with (out/"attempts.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(attempts)
            return episode
        except Exception as exc:
            errors.append({"component":"rollout","robot":robot,"teacher_origin":origin,"seed":seed,"error":str(exc),"traceback":traceback.format_exc()})
            _write(out/"exceptions.json",errors)
            print(f"[stage3] rollout failed {origin}: {exc}",flush=True)
            return None
    seed=int(config.get("seed",42))
    execute(seed,"source_scripted",robot="Panda",video=videos/"source.mp4")
    goal=int(config.get("target_teacher_success_goal",3))
    cap=int(config.get("target_attempt_cap",20))
    transfer_cap=min(cap,int(config.get("source_retarget_attempt_cap",max(goal*2,6)))) if traces else 0
    transfer_successes=0
    target_attempts=0
    for i in range(transfer_cap):
        trace=traces[i%len(traces)]
        ep=execute(seed+1000+i,"source_retargeted",trace=trace,
                   video=videos/"target_teacher.mp4" if i==0 else None)
        target_attempts+=1
        transfer_successes+=int(bool(ep and ep["success"]))
        if transfer_successes>=goal: break
    fallback_successes=0
    while transfer_successes+fallback_successes<goal and target_attempts<cap:
        i=target_attempts
        ep=execute(seed+1000+i,"target_scripted_bootstrap",
                   video=videos/"target_teacher_scripted.mp4" if fallback_successes==0 else None)
        target_attempts+=1
        fallback_successes+=int(bool(ep and ep["success"]))
        if ep and ep["success"] and not (videos/"target_teacher.mp4").exists():
            import shutil
            shutil.copyfile(videos/"target_teacher_scripted.mp4",videos/"target_teacher.mp4")
    counts={origin:{"attempts":sum(ep["teacher_origin"]==origin for ep in episodes),
                    "successes":sum(ep["teacher_origin"]==origin and ep["success"] for ep in episodes)}
            for origin in ("source_scripted","source_retargeted","target_scripted_bootstrap")}
    stats={**status,"status":"completed" if transfer_successes else "completed_degraded",
           "duration_seconds":time.monotonic()-started,"episodes":episodes,"counts":counts,
           "source_dataset_path":str(dataset),"selected_source_demonstration_ids":ids,
           "source_trace_count":len(traces),"target_attempts":target_attempts,
           "target_teacher_success_goal":goal,"target_teacher_success_count":transfer_successes+fallback_successes,
           "genuine_transfer_verified":bool(transfer_successes),"source_retargeted_successes":transfer_successes,
           "target_scripted_successes":fallback_successes,"errors":errors,
           "input_task_path":str(task_path),"task_source":task.get("task_source","provided_seed"),
           "policy_transfer_parameters":next((ep["transfer_parameters"] for ep in episodes if ep["robot"]=="UR5e" and ep["success"]),None),
           "teacher_data_limit": "source-relative waypoint extraction plus privileged target feedback; no learned source policy"}
    stats["policy_tracker_parameters"] = stats["policy_transfer_parameters"]
    _write(out/"stats.json",stats)
    _write(out/"status.json",{k:v for k,v in stats.items() if k not in ("episodes","errors")})
    if errors: _write(out/"exceptions.json",errors)
    return stats


def main():
    import yaml
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/bootstrap.yaml");parser.add_argument("--run-dir",required=True)
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(run(config,Path(args.run_dir)),indent=2))

if __name__=="__main__": main()
