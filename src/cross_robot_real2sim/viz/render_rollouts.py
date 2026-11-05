from pathlib import Path
import numpy as np


def annotate(frame, label):
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(frame).astype("uint8"))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0,0,im.width,33), fill="black")
    # Keep labels readable on 224-pixel evidence frames.
    words, lines, line = label.split(), [], ""
    for word in words:
        if len(line)+len(word)>35:
            lines.append(line); line = ""
        line += word+" "
    lines.append(line)
    for i, line in enumerate(lines[:3]):
        draw.text((3,2+10*i), line, fill="white")
    return np.asarray(im)


def write_video(frames, path, fps=10, label=None):
    import imageio.v2 as imageio
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(annotate(frame,label) if label else frame)
    return str(path)
