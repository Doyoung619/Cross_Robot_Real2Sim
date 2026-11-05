from pathlib import Path
import argparse
import time
import traceback
from cross_robot_real2sim.common.config import load_config,resolve_path
from cross_robot_real2sim.common.io import write_json
from cross_robot_real2sim.common.provenance import provenance
from cross_robot_real2sim.adapters.source_hdf5 import ensure_dataset,inspect_dataset
from cross_robot_real2sim.schemas import Selection
from .metadata_baseline import select_demonstrations


def _run(config,run_dir):
    start=time.monotonic();out=Path(run_dir)/"01_selection";out.mkdir(parents=True,exist_ok=True)
    source=ensure_dataset(config)
    inspection=inspect_dataset(source,config.get("source_demo_limit",3));write_json(out/"source_inspection.json",{**provenance(config,"hdf5_schema_inspector",[str(source)]),**inspection})
    selected,candidates=select_demonstrations(source,config)
    human_dir=resolve_path(config.get("human_video_dir","data/human"),config.get("repo_root"))
    clips=sorted(human_dir.glob("*.mp4"))+sorted(human_dir.glob("*.MP4"))
    if len(clips)>int(config.get("human_clip_cap",8)): raise ValueError("Human clip count exceeds compute budget")
    ranking=[];chosen=[]
    human_status="skipped_missing_input" if not clips else "skipped_bootstrap_mode"
    strategy="metadata_consistency_shortest_then_diverse_start"
    if config.get("mode")=="full":
        if clips:
            from .clip_ranker import rank_clips
            ranking=rank_clips(clips,config.get("task_text","place the can inside the bin"),config,out/"sampled_frames")
            chosen=[x["path"] for x in ranking if x["selected"]]
            human_status="completed";strategy+="+CLIP_human_video_cosine"
        else: human_status="requires_human_video"
    result={**provenance(config,"metadata_selector" if not ranking else "metadata_selector+openai/clip-vit-base-patch32",
                          [f"robomimic:{x['demo_id']}" for x in selected]),
            "status":"completed" if human_status=="completed" else "completed_degraded",
            "selected_human_clip_paths":chosen,"selected_source_demonstration_ids":[x["demo_id"] for x in selected],
            "source_dataset_path":str(source.resolve()),"source_dataset_sha256":inspection["sha256"],
            "source_dataset_id":"robomimic/robomimic_datasets/v1.5/can/ph/low_dim_v15.hdf5",
            "human_clip_selection":human_status,"human_video_candidates":[str(x.resolve()) for x in clips],
            "source_candidates":candidates,"selected_source_metadata":selected,"clip_ranking":ranking,
            "selection_strategy":strategy,"source_policy_type":"human_demonstration_dataset",
            "reason":"Source metadata selection executed; absent/unrequested human perception is explicitly skipped" if not ranking else "Real human frames ranked by CLIP",
            "duration_seconds":time.monotonic()-start}
    Selection.model_validate(result)
    write_json(out/"selection.json",result);write_json(out/"status.json",{**provenance(config,"metadata_selector",result["input_ids"]),**{k:result[k] for k in ("status","reason","duration_seconds")}})
    return result


def run(config,run_dir):
    try:
        return _run(config,run_dir)
    except Exception as exc:
        write_json(Path(run_dir)/"01_selection/status.json",{
            **provenance(config,"stage1_selection",[]),"status":"failed",
            "reason":str(exc),"exception_type":type(exc).__name__,"traceback":traceback.format_exc()
        })
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/bootstrap.yaml");p.add_argument("--run-dir",required=True)
    a=p.parse_args();run(load_config(a.config),Path(a.run_dir))

if __name__=="__main__":main()
