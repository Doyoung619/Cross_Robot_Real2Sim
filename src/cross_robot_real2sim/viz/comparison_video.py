"""Evidence montage; missing human input is visibly identified."""
from pathlib import Path
import json
import numpy as np


def create_comparison(run_dir):
    import imageio.v2 as imageio
    from PIL import Image
    from .render_rollouts import annotate, write_video
    root = Path(run_dir)
    stats_path=root/"03_retarget/stats.json"
    stats=json.loads(stats_path.read_text()) if stats_path.exists() else {}
    teacher=next((ep for ep in stats.get("episodes",[]) if ep.get("robot")=="UR5e" and Path(ep.get("video_path","missing")).name=="target_teacher.mp4"),{})
    teacher_label=f"UR5e {teacher.get('teacher_origin','teacher')}"
    paths = [("Panda scripted source",root/"videos/source.mp4"),
             (teacher_label,root/"videos/target_teacher.mp4"),
             ("UR5e learned BC",root/"videos/target_bc.mp4")]
    selection_path=root/"01_selection/selection.json"
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    human = [Path(p) for p in selection.get("selected_human_clip_paths",[]) if Path(p).is_file()]
    if human:
        paths.insert(0,("Real human input",human[0]))
    clips = []
    def letterbox(frame):
        im=Image.fromarray(frame)
        im.thumbnail((224,224))
        canvas=Image.new("RGB",(224,224),"black")
        canvas.paste(im,((224-im.width)//2,(224-im.height)//2))
        return np.asarray(canvas)
    for label,path in paths:
        if path.exists():
            clips.append((label,[letterbox(f) for f in imageio.mimread(path)]))
    if not clips:
        return None
    count = max(len(frames) for _,frames in clips)
    frames_out=[]
    for i in range(count):
        panels=[]
        if not human:
            panels.append(annotate(np.zeros((224,224,3),dtype=np.uint8),"NO HUMAN VIDEO SUPPLIED"))
        for label,frames in clips:
            panels.append(annotate(frames[min(i,len(frames)-1)],label))
        frames_out.append(np.concatenate(panels,axis=1))
    return write_video(frames_out,root/"videos/comparison.mp4",fps=10)


# Public alias retained for callers of the original orchestrator interface.
make_comparison=create_comparison
