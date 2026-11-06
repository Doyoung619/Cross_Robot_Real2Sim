"""English-only videos tracing original A data -> executed B teachers -> shared B policy."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from ..common.provenance import file_sha256

SIZE=(1920,1080)
FPS=20
BG=(13,21,36)


@lru_cache(maxsize=24)
def font(size,bold=False):
    name="DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}",size)


def lines(draw,text,typeface,width):
    result=[];line=""
    for word in text.split():
        candidate=(line+" "+word).strip()
        if line and draw.textlength(candidate,font=typeface)>width:
            result.append(line);line=word
        else:line=candidate
    if line:result.append(line)
    return result


class EnglishMovie:
    def __init__(self,path):
        import imageio.v2 as imageio
        self.path=path
        self.writer=imageio.get_writer(str(path),fps=FPS,codec="libx264",quality=7,macro_block_size=1)
        self.frame_count=0;self.chapters=[];self.text=[]

    def base(self,title,subtitle):
        self.text.extend([title,subtitle])
        canvas=Image.new("RGB",SIZE,BG);draw=ImageDraw.Draw(canvas)
        draw.text((40,28),title,font=font(40,True),fill="white")
        draw.text((42,90),subtitle,font=font(25),fill=(255,211,132))
        return canvas

    def append(self,canvas):
        self.writer.append_data(np.asarray(canvas));self.frame_count+=1

    def card(self,title,subtitle,paragraphs,seconds=8):
        start=self.frame_count
        canvas=self.base(title,subtitle);draw=ImageDraw.Draw(canvas);y=175
        for paragraph in paragraphs:
            self.text.append(paragraph)
            for line in lines(draw,paragraph,font(31),1800):
                draw.text((50,y),line,font=font(31),fill=(228,235,247));y+=47
            y+=24
        if y>1030:raise ValueError(f"Text overflow in {title}: bottom={y}")
        for _ in range(round(seconds*FPS)):self.append(canvas)
        self.chapters.append({"title":title,"start_seconds":start/FPS,"end_seconds":self.frame_count/FPS})

    def training_card(self,case,manifest):
        start=self.frame_count
        canvas=self.base("What data trained Robot B?", "ONE shared checkpoint across all three cases; no source-only policy was trained.")
        draw=ImageDraw.Draw(canvas)
        y=185
        headline=f"{manifest['training_teacher_episodes']} successful executed UR5e episodes | {manifest['training_state_action_samples']} state-action pairs | {manifest['training_steps']} Adam updates"
        draw.text((45,y),headline,font=font(30,True),fill="white");self.text.append(headline)
        groups={}
        for row in manifest["training_lineage"]:
            groups.setdefault(row["source_demo_id"],[]).append(row)
        y=280
        draw.text((50,y),"Original Panda demo",font=font(28,True),fill=(180,199,224))
        draw.text((530,y),"Executed UR5e training seeds",font=font(28,True),fill=(180,199,224))
        draw.text((1330,y),"Samples",font=font(28,True),fill=(180,199,224));y+=68
        for demo_id,rows in sorted(groups.items()):
            selected=demo_id==case["source_demo_id"]
            if selected:draw.rounded_rectangle((35,y-9,1855,y+59),radius=9,fill=(28,79,87))
            color=(115,250,192) if selected else (225,232,244)
            seeds=" + ".join(str(row["target_teacher_seed"]) for row in rows)
            samples=" + ".join(str(row["training_samples"]) for row in rows)
            draw.text((50,y),demo_id+("  [shown in this case]" if selected else ""),font=font(28),fill=color)
            draw.text((530,y),seeds,font=font(28),fill=color)
            draw.text((1330,y),samples,font=font(28),fill=color)
            self.text.append(f"{demo_id}: teacher seeds {seeds}; samples {samples}")
            y+=83
        footer=["BC input: 26 state features, including observable phase; output: XYZ delta and gripper.",
                "Architecture: 26 -> 128 ReLU -> 128 ReLU -> 4. Inference uses oracle_sim_state.",
                "Checkpoint SHA256: "+manifest["shared_checkpoint_sha256"],
                "The highlighted source contributes two teacher episodes; all ten episodes train the SAME policy."]
        y=815
        for text in footer:
            self.text.append(text);draw.text((45,y),text,font=font(24),fill=(255,215,146));y+=48
        for _ in range(9*FPS):self.append(canvas)
        self.chapters.append({"title":"Training data lineage","start_seconds":start/FPS,"end_seconds":self.frame_count/FPS})

    def scene(self,title,subtitle,panels,notes):
        import imageio.v2 as imageio
        start=self.frame_count;opened=[]
        try:
            for label,path in panels:
                reader=imageio.get_reader(path);meta=reader.get_meta_data();count=reader.count_frames();fps=float(meta["fps"])
                opened.append({"label":label,"path":path,"reader":reader,"frames":count,"fps":fps,
                               "duration":count/fps,"last_index":None,"last_image":None})
                self.text.append(label)
            duration=max(item["duration"] for item in opened)
            for out_index in range(math.ceil(duration*FPS)):
                t=out_index/FPS
                canvas=self.base(title,subtitle);draw=ImageDraw.Draw(canvas)
                area_w=1840/len(opened);area_h=700 if len(opened)==1 else 560
                for index,item in enumerate(opened):
                    frame_index=min(int(t*item["fps"]+1e-9),item["frames"]-1)
                    if frame_index!=item["last_index"]:
                        item["last_image"]=Image.fromarray(item["reader"].get_data(frame_index));item["last_index"]=frame_index
                    image=item["last_image"]
                    scale=min((area_w-14)/image.width,area_h/image.height)
                    frame=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
                    x=40+index*area_w+(area_w-frame.width)/2
                    y=190+(area_h-frame.height)/2
                    canvas.paste(frame,(round(x),round(y)))
                    draw.text((round(45+index*area_w),147),item["label"],font=font(23,True),fill="white")
                    if t>=item["duration"]:
                        draw.rectangle((round(x),round(y+frame.height-34),round(x+frame.width),round(y+frame.height)),fill="black")
                        draw.text((round(x+8),round(y+frame.height-31)),"EPISODE ENDED - final recorded frame held",font=font(18),fill=(255,217,143))
                note_y=925 if len(opened)==1 else 820
                for note in notes:
                    if out_index==0:self.text.append(note)
                    for line in lines(draw,note,font(24),1810):
                        draw.text((45,note_y),line,font=font(24),fill=(222,232,247));note_y+=38
                    note_y+=8
                if note_y>1070:raise ValueError(f"Scene notes overflow in {title}")
                self.append(canvas)
            self.chapters.append({"title":title,"start_seconds":start/FPS,"end_seconds":self.frame_count/FPS,
                                  "source_videos":[str(item["path"]) for item in opened],"timing":"Each panel preserves its own original video FPS; independent starts; no temporal correspondence implied"})
        finally:
            for item in opened:item["reader"].close()

    def close(self):
        self.writer.close()
        text="\n".join(dict.fromkeys(self.text))
        if not text.isascii():raise ValueError("Non-ASCII overlay text found in English-only case video")
        text_path=self.path.with_suffix(".txt");text_path.write_text(text+"\n")
        return {"path":str(self.path),"frames":self.frame_count,"fps":FPS,"duration_seconds":self.frame_count/FPS,
                "width":SIZE[0],"height":SIZE[1],"chapters":self.chapters,"overlay_transcript":str(text_path),
                "overlay_language":"English","video_sha256":file_sha256(self.path)}


def assemble_cases(root:Path,output:Path,manifest:dict):
    import csv
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    videos=output/"videos";videos.mkdir(exist_ok=True)
    with (output/"training_lineage.csv").open() as handle:manifest["training_lineage"]=list(csv.DictReader(handle))
    movies=[]
    for case in manifest["cases"]:
        source=json.loads((output/"source"/f"source_{case['source_demo_id']}.json").read_text())
        source_path=output/case["source_video"]
        if file_sha256(source_path)!=source["video_sha256"]:raise ValueError("Source video hash mismatch")
        if source["max_eef_position_error_m"]>1e-8 or source["max_can_position_error_m"]>1e-8:
            raise ValueError("Original Robot A state reconstruction does not match source observations")
        case["source_replay"]=source
        teacher=case["training_teacher"];learner=case["learned_B"]
        teacher_path=output/teacher["video_path"];learner_path=output/learner["video_path"]
        for replay,path in [(teacher,teacher_path),(learner,learner_path)]:
            if replay.get("video_language")!="en" or file_sha256(path)!=replay["video_sha256"]:
                raise ValueError("Target replay language/hash check failed")
        movie=EnglishMovie(output/case["case_video"])
        try:
            movie.card(f"Case {case['case_id']}: original Panda {case['source_demo_id']} to UR5e",
                       "Original A motion, executed B training data, and a successful held-out B rollout",[
                "Robot A: recorded Panda demonstration reconstructed from the original HDF5 simulator states and XML.",
                f"Robot B teacher: seed {teacher['seed']}, using grasp/lift geometry extracted from {case['source_demo_id']}.",
                f"Robot B learned policy: held-out seed {learner['seed']}; SAME frozen checkpoint in all three cases.",
                "The learned B policy was trained on ten executed UR5e demonstrations derived from five Panda demos.",
                "No human MP4 was supplied. This is a fixed-task simulation baseline using oracle_sim_state."],seconds=9)
            movie.scene("Robot A: what the original demonstration actually did",
                        "Recorded dataset motion; state reconstruction is not a new scripted policy or action re-simulation.",
                        [(f"Original Panda {case['source_demo_id']} | {source['validated_frames']} original frames",source_path)],
                        [f"All original frames checked: Can error {source['max_can_position_error_m']:.1e} m; EEF error <= {source['max_eef_position_error_m']:.2e} m.",
                         f"Original 20 Hz timing preserved; episode duration {source['video_duration_s']:.2f} s; original scene/camera retained."])
            movie.training_card(case,manifest)
            original_teacher_path=root/teacher["cache_key"]["request"]["original_npz"]
            with np.load(original_teacher_path) as data:original_teacher=json.loads(str(data["metadata_json"].item()))
            parameters=original_teacher["transfer_parameters"]
            offset_text=", ".join(f"{1000*x:.1f}" for x in parameters["grasp_offset"])
            movie.scene("Robot B: the executed synthetic demonstration used for training",
                        "Source-derived grasp offset and lift clearance + hand-coded phase controller + target OSC feedback",
                        [(f"Training teacher seed {teacher['seed']} | source {case['source_demo_id']}",teacher_path)],
                        [f"Transferred grip offset [x,y,z] = [{offset_text}] mm; target lift clearance = {1000*parameters['clearance']:.1f} mm.",
                         f"This actual training episode contains {teacher['terminal_step']} state-action pairs. Its matching source contributes a second teacher episode too."])
            movie.scene("Robot B: how the learned policy operates",
                        "Frozen shared MLP: 26D state + online phase -> normalized XYZ delta and gripper -> OSC adapter",
                        [(f"Held-out evaluation seed {learner['seed']} | shared BC checkpoint",learner_path)],
                        ["The source demo is not an input at inference. This rollout was not used for training.",
                         "After the original endpoint, the SAME learned policy continues for one second to verify stable placement."])
            movie.scene("Original A motion / B training demonstration / learned B rollout",
                        "Independent initial states and time axes. No frame-by-frame path or timing correspondence is claimed.",
                        [(f"A: original {case['source_demo_id']}",source_path),(f"B: teacher seed {teacher['seed']}",teacher_path),(f"B: learned seed {learner['seed']}",learner_path)],
                        ["The source contributes grasp/lift geometry. The target movement is generated by its own feedback controller.",
                         "The learned B policy is the same pooled-data model in all three case videos.",
                         "Retargeting scope: a constrained geometry-based baseline; causal benefit over source-free control is untested."])
            movie.card("Measured outcomes and claim boundaries", "Success means executed placement, not visual resemblance alone",[
                f"Original Robot A reconstruction: all {source['validated_frames']} frames verified against source observations.",
                f"B teacher: original and replay success=True; 1-second stable placement={teacher['stable_placement_1s']}; terminal XY error={1000*teacher['terminal']['xy_error_m']:.2f} mm.",
                f"B learned: original and replay success=True; 1-second stable placement={learner['stable_placement_1s']}; terminal XY error={1000*learner['terminal']['xy_error_m']:.2f} mm.",
                "One shared checkpoint trained on ten target demonstrations / 1,620 state-action pairs. No new training for these videos.",
                "Human-video inference, full source pose/time-trajectory transfer, source-free ablation and real-robot transfer remain untested.",
                "See the English technical PDF and case_manifest.json for data lineage, methods, failures and reproducibility."],seconds=8)
        finally:movies.append(movie.close())
    media=[]
    for path in sorted(output.rglob("*.mp4")):
        frames,duration=count_frames_and_secs(str(path));reader=imageio.get_reader(path)
        try:
            first,last=reader.get_data(0),reader.get_data(frames-1)
            if first.shape!=last.shape or not frames:raise ValueError(f"Invalid video {path}")
        finally:reader.close()
        media.append({"path":str(path.relative_to(output)),"frames":frames,"duration_seconds":duration,
                      "frame_shape":list(first.shape),"bytes":path.stat().st_size,"sha256":file_sha256(path)})
    manifest["movies"]=movies;manifest["verified_media"]=media
    (output/"case_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"[english cases] assembled {len(movies)} case movies and verified {len(media)} English MP4 assets",flush=True)
