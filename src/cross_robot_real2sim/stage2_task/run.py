from pathlib import Path
import argparse
import time
import traceback
from cross_robot_real2sim.common.config import load_config
from cross_robot_real2sim.common.io import read_json,write_json
from cross_robot_real2sim.common.provenance import provenance
from cross_robot_real2sim.schemas import TaskSpec
from .seed_baseline import load_seed
from .sam2_optional import segmentation_status


def _run(config,run_dir):
    start=time.monotonic();run_dir=Path(run_dir);out=run_dir/"02_task";out.mkdir(parents=True,exist_ok=True)
    selection=read_json(run_dir/"01_selection/selection.json")
    clips=selection.get("selected_human_clip_paths",[])
    if config.get("mode")=="full":
        if not clips:
            result={**provenance(config,"Qwen/Qwen2.5-VL-3B-Instruct",[]),"status":"skipped_missing_input",
                    "reason":"requires_human_video","task_source":None,"perception_status":"requires_human_video",
                    "duration_seconds":time.monotonic()-start}
            write_json(out/"status.json",result)
            # Remove stale success metadata if a previously supplied clip was removed on resume.
            if (out/"task_spec.json").exists():(out/"task_spec.json").unlink()
            return result
        from .qwen_vl import infer_task,MODEL_ID
        task=infer_task(clips[0],config,out)
        input_ids=clips;model=MODEL_ID;status="completed";reason="Actual human RGB frames interpreted by local Qwen"
        perception_status="completed"
    else:
        task,seed_path=load_seed(config)
        input_ids=[seed_path];model="provided_task_seed";status="completed_degraded"
        reason="Provided task seed validated; human-video interpretation not run"
        perception_status="skipped_missing_input" if not selection.get("human_video_candidates") else "skipped_bootstrap_mode"
    result={**task,**provenance(config,model,input_ids,privileged_state_used=True),"status":status,"reason":reason,
            "perception_status":perception_status,"segmentation":segmentation_status(config),
            "goal_pose_source":"simulator_oracle","duration_seconds":time.monotonic()-start}
    TaskSpec.model_validate(result)
    write_json(out/"task_spec.json",result)
    write_json(out/"status.json",{**provenance(config,model,input_ids,privileged_state_used=True),**{k:result[k] for k in ("status","reason","perception_status","duration_seconds")}})
    return result


def run(config,run_dir):
    try:
        return _run(config,run_dir)
    except Exception as exc:
        write_json(Path(run_dir)/"02_task/status.json",{
            **provenance(config,"stage2_task",[]),"status":"failed",
            "reason":str(exc),"exception_type":type(exc).__name__,"traceback":traceback.format_exc()
        })
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/bootstrap.yaml");p.add_argument("--run-dir",required=True)
    a=p.parse_args();run(load_config(a.config),Path(a.run_dir))

if __name__=="__main__":main()
