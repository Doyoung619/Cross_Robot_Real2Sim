"""Real CLIP ranking, optional until full mode. Model references are released on return."""
from pathlib import Path
import gc
import numpy as np
from PIL import Image, ImageDraw

MODEL_ID="openai/clip-vit-base-patch32"


def sample_frames(path, output_dir=None, count=6, max_seconds=120, max_bytes=200_000_000):
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    path=Path(path)
    if path.stat().st_size>max_bytes: raise ValueError(f"Clip exceeds {max_bytes} byte budget: {path}")
    nframes,duration=count_frames_and_secs(str(path))
    if duration>max_seconds: raise ValueError(f"Clip exceeds {max_seconds} second budget: {path}")
    if nframes<1: raise ValueError(f"No frames decoded: {path}")
    count=min(max(4,int(count)),6,nframes)
    indices=np.unique(np.linspace(0,nframes-1,count,dtype=int))
    reader=imageio.get_reader(str(path))
    frames=[];records=[]
    try:
        for i,index in enumerate(indices):
            frame=Image.fromarray(reader.get_data(int(index))).convert("RGB")
            frame.thumbnail((336,336),Image.Resampling.LANCZOS)
            timestamp=float(index*duration/max(nframes,1))
            frames.append(frame)
            record={"frame_index":int(index),"timestamp_seconds":timestamp,"frame":"human_image_px","input_type":"human_video"}
            if output_dir:
                output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
                raw_path=output/f"frame_{i:02d}.jpg";frame.save(raw_path)
                annotated=frame.copy();draw=ImageDraw.Draw(annotated)
                draw.rectangle((0,0,annotated.width,25),fill="black")
                draw.text((3,3),f"HUMAN VIDEO  t={timestamp:.2f}s",fill="white")
                annotated.save(output/f"annotated_{i:02d}.jpg")
                record["path"]=str(raw_path.resolve())
            records.append(record)
    finally: reader.close()
    return frames,records


def rank_clips(clips,task_text,config,output_dir):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device=config.get("device","cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for full-mode CLIP but unavailable")
    model=CLIPModel.from_pretrained(MODEL_ID).eval().to(device)
    processor=CLIPProcessor.from_pretrained(MODEL_ID)
    results=[]
    try:
        for index,clip in enumerate(clips):
            frames,records=sample_frames(clip,Path(output_dir)/f"clip_{index:02d}",config.get("video_frame_count",6))
            inputs=processor(text=[task_text],images=frames,return_tensors="pt",padding=True).to(device)
            with torch.inference_mode():
                output=model(**inputs)
                cosine=(output.logits_per_image/model.logit_scale.exp()).float().cpu().numpy()[:,0]
            results.append({"path":str(Path(clip).resolve()),"cosine_similarity":float(cosine.mean()),
                            "frame_scores":cosine.tolist(),"frames":records,"model_name":MODEL_ID})
            del inputs,output
    finally:
        del model,processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    results.sort(key=lambda r:(-r["cosine_similarity"],r["path"]))
    keep=int(config.get("selected_human_clip_limit",1))
    for i,result in enumerate(results):
        result["rank"]=i+1;result["selected"]=i<keep
        result["reason"]="highest_real_CLIP_task_similarity" if i<keep else "ranked_below_clip_selection_budget"
    return results
