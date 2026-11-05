"""Sequential local Qwen inference on actual ordered human frames.

API follows https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct model card,
using installed Transformers 4.x Qwen2_5_VLForConditionalGeneration.
No weights are loaded at import or during bootstrap.
"""
from pathlib import Path
import gc
import json
from cross_robot_real2sim.common.io import write_json
from cross_robot_real2sim.schemas import TaskSpec
from cross_robot_real2sim.stage1_selection.clip_ranker import sample_frames

MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"


def parse_json_object(text):
    decoder=json.JSONDecoder()
    for offset,char in enumerate(text):
        if char=="{":
            try:
                value,_=decoder.raw_decode(text[offset:])
                if isinstance(value,dict):return value
            except json.JSONDecodeError:pass
    raise ValueError("Qwen output contains no parseable JSON object")


def infer_task(clip,config,out):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    from qwen_vl_utils import process_vision_info
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device=config.get("device","cuda:0")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Full Qwen path requires configured GPU 0 and working CUDA")
    frames,records=sample_frames(clip,out/"annotated_frames",config.get("video_frame_count",6))
    del frames
    instruction='''The images are ordered frames from one real human demonstration. Identify what actually happens; do not assume the requested task occurred. Return exactly one JSON object with keys object (lowercase object label), goal (object with relation and receptacle), stages (ordered action strings), constraints (strings), unknowns ({mass:"unknown",friction:"unknown",contact_state:"unknown"}). Do not infer metric 3D motion, material friction, mass, or proven contact from RGB. Use relation "inside" only if supported by the frames; say "unknown" when unclear. Task under study is picking, lifting, transporting, placing a can inside a bin; the observed goal must still be grounded in these images.'''
    content=[]
    for record in records:
        content.extend([{"type":"text","text":f"Ordered frame at {record['timestamp_seconds']:.3f} seconds:"},
                        {"type":"image","image":record["path"],"max_pixels":336*336}])
    content.append({"type":"text","text":instruction})
    messages=[{"role":"user","content":content}]
    write_json(out/"qwen_input.json",{"clip":str(Path(clip).resolve()),"ordered_frames":records,"prompt":instruction,
                                      "model_id":MODEL_ID,"input_type":"human_video"})
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID,torch_dtype=dtype,attn_implementation="sdpa").eval().to(device)
    processor=AutoProcessor.from_pretrained(MODEL_ID,min_pixels=56*56,max_pixels=336*336)
    inputs=generated=trimmed=images=videos=None
    try:
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,videos=process_vision_info(messages)
        inputs=processor(text=[text],images=images,videos=videos,padding=True,return_tensors="pt").to(device)
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=int(config.get("qwen_max_new_tokens",512)),do_sample=False)
        trimmed=[out_ids[len(in_ids):] for in_ids,out_ids in zip(inputs.input_ids,generated)]
        raw=processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        (out/"qwen_raw_output.txt").write_text(raw)
    finally:
        del model,processor,inputs,generated,trimmed,images,videos
        gc.collect();torch.cuda.empty_cache()
    parsed=parse_json_object(raw)
    write_json(out/"qwen_parsed_unmapped.json",parsed)
    # Deliberately reject unknown/unrelated goals; never coerce the observation into this benchmark.
    if parsed.get("object")!="can" or parsed.get("goal")!={"relation":"inside","receptacle":"bin"}:
        raise ValueError(f"goal_parsing_or_source_selection_mismatch: observed {parsed.get('object')} -> {parsed.get('goal')}")
    parsed.update({"schema_version":"1.0","task_id":"can_to_bin","task_source":"human_video_qwen",
                   "goal_pose_source":"simulator_oracle",
                   "unknowns":{"mass":"unknown","friction":"unknown","contact_state":"unknown"},
                   "semantic_to_simulator":{"can":"Can","bin":"PickPlaceCan.target_bin_placements[can_index]"},
                   "assumptions":["Observed semantic can maps to robosuite Can",
                                  "Observed semantic bin maps to assigned Can bin; visual appearance need not match",
                                  "Metric target pose uses privileged simulator state; this is not vision-only transfer"],
                   "human_clip_path":str(Path(clip).resolve()),"frame_records":records,"model_name":MODEL_ID})
    return TaskSpec.model_validate(parsed).model_dump()
