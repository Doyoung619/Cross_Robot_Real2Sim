#!/usr/bin/env python3
"""English side-by-side movies of untouched final feedback-test scenes."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def font(size,bold=False):
    suffix='-Bold' if bold else ''
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+suffix+'.ttf',size)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];raw=Path(args.run_dir).resolve()
    status=json.loads((raw/'status.json').read_text())
    if status['status']!='completed':raise RuntimeError('Wait for automatic packaging to finish first')
    dest=root/'results'/raw.name;(dest/'videos').mkdir(exist_ok=True)
    rows=json.loads((raw/'final/evaluation.json').read_text())
    videos=[]
    for demo in ['demo_47','demo_48']:
        cases=[next(r for r in rows if r['condition']==condition and r['source_initial_scene_id']==demo)
               for condition in ['final_original100','final_selected_round_1']]
        if cases[0]['initial_plate_position']!=cases[1]['initial_plate_position']:
            raise ValueError('Final comparison scene mismatch')
        paths=[raw/'final'/r['video'] for r in cases]
        readers=[imageio.get_reader(p) for p in paths]
        lengths=[r.count_frames() for r in readers]
        output=dest/'videos'/f'final_comparison_{demo}.mp4'
        try:
            with imageio.get_writer(str(output),fps=10,codec='libx264',quality=8,macro_block_size=1) as writer:
                for i in range(max(lengths)):
                    canvas=Image.new('RGB',(1280,850),(12,20,32));draw=ImageDraw.Draw(canvas)
                    draw.text((24,22),'Reserved final test | original B vs feedback-selected B',font=font(28,True),fill='white')
                    draw.text((24,68),f'{demo} | same starting scene | not used to select source data or choose the round',font=font(21),fill=(180,213,241))
                    labels=['Original adapter: 100 updates','Selected feedback round 1: 200 total updates']
                    for j,(reader,case,label) in enumerate(zip(readers,cases,labels)):
                        draw.text((640*j+20,115),label,font=font(21,True),fill='white')
                        canvas.paste(Image.fromarray(reader.get_data(min(i,lengths[j]-1))),(640*j,155))
                        outcome='SUCCESS' if case['success'] else 'FAILURE'
                        draw.text((640*j+24,545),outcome,font=font(30,True),fill=(115,225,160) if case['success'] else (255,180,120))
                        draw.text((640*j+24,591),f"Final goal error: {case['goal_error_m']*100:.2f} cm",font=font(23),fill='white')
                        draw.text((640*j+24,628),f"Plate displacement: {case['plate_displacement_m']*100:.2f} cm",font=font(21),fill='white')
                        if i>=lengths[j]:
                            draw.text((640*j+24,674),'Rollout ended; terminal frame held',font=font(19),fill=(180,213,241))
                    draw.text((24,727),'Real B policy + physics execution. No scripted trajectory override.',font=font(23),fill='white')
                    draw.text((24,770),'Both reserved final scenes are shown, including the unsuccessful feedback case.',font=font(22),fill=(255,210,150))
                    writer.append_data(np.asarray(canvas))
        finally:
            for reader in readers:reader.close()
        with imageio.get_reader(output) as reader:
            count=reader.count_frames();metadata=reader.get_meta_data()
        assert count==max(lengths)
        videos.append(dict(path=str(output.relative_to(dest)),frames=count,fps=metadata['fps'],
            bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            source_initial_scene_id=demo,original_success=cases[0]['success'],feedback_success=cases[1]['success'],
            original_goal_error_m=cases[0]['goal_error_m'],feedback_goal_error_m=cases[1]['goal_error_m']))
    addition=('\n\n## Reserved final before/after comparison videos\n\n'
              '- [Final demo_47: original 100 updates vs selected feedback round 1](videos/final_comparison_demo_47.mp4)\n'
              '- [Final demo_48: original 100 updates vs selected feedback round 1](videos/final_comparison_demo_48.mp4)\n\n'
              'These show both reserved scenes, with the same initial states. Successful rollouts end after twenty consecutive successful steps; their terminal image is held when the other panel continues. No failed case is omitted.\n')
    for name in ['README.md','FINAL_REPORT.md']:
        path=dest/name;text=path.read_text().split('\n\n## Reserved final before/after comparison videos')[0]
        path.write_text(text+addition)
    (dest/'videos/final_comparison_metadata.json').write_text(json.dumps(videos,indent=2))
    print(json.dumps(videos,indent=2))


if __name__=='__main__':main()
