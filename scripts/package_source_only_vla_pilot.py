#!/usr/bin/env python3
"""Export measured source-only pilot results, English videos and a compact PDF."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]


def font(size,bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',size)


def comparison(raw,dest,rows,demo):
    cases=[next(r for r in rows if r['condition']==c and r['source_initial_scene_id']==demo)
           for c in ['A_pretrained_Panda','B_pretrained_UR5e','B_adapted_UR5e']]
    readers=[imageio.get_reader(raw/r['video']) for r in cases]
    lengths=[r.count_frames() for r in readers]
    output=dest/'videos'/f'comparison_{demo}.mp4'
    try:
        with imageio.get_writer(str(output),fps=10,codec='libx264',quality=8,macro_block_size=1) as writer:
            for i in range(max(lengths)):
                canvas=Image.new('RGB',(1920,1080),(12,20,32));d=ImageDraw.Draw(canvas)
                d.text((45,32),'Source-only VLA transfer | Panda to UR5e',font=font(40,True),fill='white')
                d.text((45,90),'Plate pushing | 3 source demonstrations | 100 LoRA updates | 0 executed B training demonstrations',font=font(25),fill=(178,213,241))
                for j,(reader,case) in enumerate(zip(readers,cases)):
                    canvas.paste(Image.fromarray(reader.get_data(min(i,lengths[j]-1))),(j*640,180))
                    outcome='SUCCESS' if case['success'] else 'FAILURE'
                    d.text((j*640+25,575),outcome,font=font(34,True),fill=(115,225,160) if case['success'] else (255,180,120))
                    d.text((j*640+25,627),f"Final goal error: {case['goal_error_m']*100:.2f} cm",font=font(25),fill='white')
                d.text((45,735),f"Held-out source scene: {demo}. Same initial object positions across all three executions.",font=font(25),fill='white')
                d.text((45,790),'B training used only offline kinematics and static renders derived from A records.',font=font(25),fill=(178,213,241))
                d.text((45,840),'Lower goal error does not count as task success. Native LIBERO success is reported above.',font=font(25),fill=(255,210,150))
                d.text((45,925),'Evaluation: RGB + language + own robot state. No oracle object/goal pose enters the policy.',font=font(23),fill='white')
                d.text((45,970),'Small feasibility test; no claim of reliable transfer or a proven data-selection benefit.',font=font(23),fill='white')
                writer.append_data(np.asarray(canvas))
        with imageio.get_reader(output) as r:
            imageio.imwrite(dest/'figures'/f'comparison_{demo}_last.png',r.get_data(r.count_frames()-1))
    finally:
        for r in readers:r.close()
    return output


def processing_video(raw,dest):
    audit=json.loads((raw/'processing_audit.json').read_text())
    selection=json.loads((raw/'selection.json').read_text())
    demo=selection['selected'][0]
    with np.load(raw/'processed_source_data.npz') as f:images=f['images'].copy()
    lookup={x['source_frame']:images[i,0] for i,x in enumerate(audit['lineage']) if x['source_demo_id']==demo}
    dataset=ROOT/'runs/vla_readiness_20261113/dataset/libero_goal/push_the_plate_to_the_front_of_the_stove_demo.hdf5'
    output=dest/'videos'/'offline_source_processing.mp4'
    with h5py.File(dataset) as handle,imageio.get_writer(str(output),fps=10,codec='libx264',quality=8,macro_block_size=1) as writer:
        group=handle['data'][demo]
        for i in range(0,len(group['actions']),2):
            canvas=Image.new('RGB',(1280,800),(12,20,32));d=ImageDraw.Draw(canvas)
            d.text((30,25),'Offline source-data processing | not a B execution demonstration',font=font(27,True),fill='white')
            left=np.asarray(group['obs/agentview_rgb'][i])[::-1].copy()
            canvas.paste(Image.fromarray(left).resize((512,512)),(30,120))
            if i in lookup:
                canvas.paste(Image.fromarray(lookup[i][::-1].copy()).resize((512,512)),(680,120))
            else:
                d.text((690,320),'Excluded by static IK/collision checks',font=font(19),fill=(255,180,120))
            d.text((30,80),f'A: original recorded Panda frame | {demo}',font=font(22),fill='white')
            d.text((680,80),'B: hypothetical pose, static rendering',font=font(22),fill='white')
            d.text((30,665),f'Source frame {i} | original time {i/20:.2f}s | target physics steps for training: 0',font=font(23),fill='white')
            d.text((30,715),'Objects follow A records. B joints come from IK. Contact dynamics are not validated.',font=font(21),fill=(255,210,150))
            writer.append_data(np.asarray(canvas))
    return output


def pdf_report(dest,rows,selection,training):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph,Table,TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    path=dest/'SOURCE_ONLY_VLA_PILOT_REPORT.pdf'
    c=canvas.Canvas(str(path),pagesize=A4,initialFontName='DejaVu',initialFontSize=10)
    width,height=A4
    style=ParagraphStyle('body',fontName='DejaVu',fontSize=10.5,leading=16,textColor=colors.HexColor('#15253a'))
    page=0;y=0
    def new_page(title):
        nonlocal y,page
        if page:c.showPage()
        page+=1;y=height-55
        c.setFont('DejaVuBold',16);c.drawString(42,y,title);y-=38
        c.setFont('DejaVu',8);c.drawRightString(width-42,25,f'Panda to UR5e | source-only VLA pilot | {page}')
    def paragraph(text):
        nonlocal y
        p=Paragraph(text,style);w,h=p.wrap(width-84,height)
        if y-h<50:new_page('Source-only VLA pilot (continued)')
        p.drawOn(c,42,y-h);y-=h+14
    new_page('Source-only VLA transfer: measured pilot')
    paragraph('Task: push the plate to the front of the stove. A is Panda (7 arm joints); B is UR5e (6 arm joints) with a Robotiq85 gripper.')
    paragraph('<b>Result: no successful B task completion was observed.</b> The adapted B policy reduced mean final XY goal error, but remained unsuccessful in both evaluated cases. A succeeded once in two trials; robust source competence is not established.')
    table=[['Condition','Success / trials','Mean XY error']]
    for condition in ['A_pretrained_Panda','B_pretrained_UR5e','B_adapted_UR5e']:
        a=[r for r in rows if r['condition']==condition]
        table.append([condition.replace('_',' '),f"{sum(x['success'] for x in a)}/{len(a)}",f"{np.mean([x['goal_error_m'] for x in a])*100:.2f} cm"])
    t=Table(table,colWidths=[252,92,105]);t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DejaVu'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9f4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#8da3b9')),('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9)]));_,h=t.wrap(width-84,height);t.drawOn(c,42,y-h);y-=h+20
    paragraph('B before adaptation: 20.47 cm mean goal error. B after adaptation: 10.95 cm. The 9.52 cm reduction (46.5%) is descriptive only: two trials, zero B successes, and no control experiment isolating selection quality.')
    paragraph('All policy inputs are two RGB cameras, the task instruction and 8D robot proprioception. Object and goal ground-truth poses are used for offline source processing and evaluation metrics, not live policy input.')
    new_page('Data selection, offline conversion and training')
    paragraph('Source materials: 50 original task demonstrations, 7,638 recorded steps. Reserve five source episodes. Inspect the remaining 45 with four-pose B IK and unintended arm-collision probes; 35 pass. Select the shortest eligible episode and then diversify initial plate positions.')
    paragraph(f"Selected source episodes: {', '.join(selection['selected'])}. They contain 396 original steps. Static per-frame checks retain 189 training anchors. Images and states of hypothetical B poses are rendered from A object records without target physics stepping.")
    paragraph('Pose conversion uses full EEF orientation and position, continuous warm-start IK, joint limits, static arm-contact checks, a fixed camera mount calibration and a source-compatible proprioception frame. OSC actions share world-frame units; the first action is corrected for residual TCP position error. Future chunk labels retain original source Cartesian commands. These are kinematic pseudo-labels, not verified successful B trajectories.')
    paragraph('The complete 450M-parameter SmolVLA checkpoint is loaded strictly from lerobot/smolvla_libero revision a24f1f380461b665158d52acd140152c79e8b36d. The visual/language backbone and original model weights are frozen. Rank-8 LoRA updates 334,592 parameters in action-expert Q/V and action projections.')
    paragraph(f"Training: {training['steps']} updates, batch four, LR 1e-4, action chunks of fifty at 20 Hz. Measured training wall time: {training['duration_seconds']:.2f} seconds, including model loading. Peak PyTorch allocated memory: {training['peak_allocated_gib']:.2f} GiB. Evaluation executes five predicted actions before replanning, with ten flow steps per query. Two reserved starts per condition have 200-step maximum horizons.")
    paragraph('Images are rendered at 256px, rotated by the official LIBERO 180-degree convention, and passed through the saved VLA image preprocessing. Normalization uses the exact saved source checkpoint tensors. An earlier run with a missing image rotation is archived locally and excluded from final comparison.')
    new_page('Observed limitations and review evidence')
    paragraph('The source VLA succeeded on demo_45 but failed on demo_46. A successful original human record does not imply the pretrained VLA will succeed. A 1/2 baseline cannot substantiate a reliable competent-source transfer claim.')
    paragraph('Both adapted B cases moved the plate closer to the goal, but remained outside native success. Offline geometry and source object motion do not supply target friction, pushing-force or contact-timing supervision. The coarse source filter and tiny adapter run leave these issues unresolved.')
    paragraph('Other limits: only three selected source episodes; 100 updates; no held-out source validation for checkpoint selection; no random-subset comparison; only two final starts; incomplete dynamic feasibility of future action chunks. Repeated source scenes were used while correcting a documented input-contract bug, so this is an engineering pilot, not a benchmark result.')
    paragraph('The English comparison movies display A, unchanged B and adapted B together for both starts, including the failed source case. offline_source_processing.mp4 distinguishes original A records from hypothetical B kinematic rendering. Raw final rollout movies and per-seed metrics accompany the report.')
    img=dest/'figures/comparison_demo_45_last.png'
    if img.exists():
        h=(width-84)*1080/1920
        if y-h<45:new_page('English video comparison still')
        c.drawImage(str(img),42,y-h,width=width-84,height=h);y-=h+12
    c.save()
    return path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');args=parser.parse_args()
    raw=Path(args.run_dir).resolve();dest=ROOT/'results'/raw.name
    dest.mkdir(parents=True,exist_ok=True)
    for folder in ['videos','figures']:(dest/folder).mkdir(exist_ok=True)
    for name in ['selection.json','selection_candidates.json','processing_audit.json','training.json','train_log.json','evaluation.json','verification.json','INPUT_CONTRACT_CORRECTION.md','status.json','adapter.safetensors']:
        if (raw/name).exists():shutil.copyfile(raw/name,dest/name)
    rows=json.loads((raw/'evaluation.json').read_text());selection=json.loads((raw/'selection.json').read_text());training=json.loads((raw/'training.json').read_text())
    for row in rows:shutil.copyfile(raw/row['video'],dest/row['video'])
    for demo in selection['evaluation_source_ids']:comparison(raw,dest,rows,demo)
    processing_video(raw,dest)
    pdf_report(dest,rows,selection,training)
    text=(raw/'FINAL_REPORT.md').read_text()
    text=text.replace('## English videos','## English comparison videos\n\n- [Comparison: demo_45](videos/comparison_demo_45.mp4)\n- [Comparison: demo_46](videos/comparison_demo_46.mp4)\n- [Offline source processing](videos/offline_source_processing.mp4)\n- [English PDF report](SOURCE_ONLY_VLA_PILOT_REPORT.pdf)\n\n## Individual final rollout videos')
    text += '\n\n## Reproduce and survive SSH disconnect\n\n```bash\nbash scripts/launch_source_only_vla_pilot.sh\n```\n\nThe detached tmux supervisor enforces a maximum of sixty minutes and records a report on failure or timeout. This run completed; no background training remains. The source materials and isolated runtime must already be installed.\n\nThe first image-contract run is archived locally and excluded. The corrected run retains all six final outcomes, including failures.\n\nThe source VLA achieved only 1/2 successes, so this pilot does not establish reliable transfer from an already competent source. Synthetic future action chunks retain A Cartesian commands; only current training anchors receive residual TCP correction and static feasibility filtering.\n'
    (dest/'FINAL_REPORT.md').write_text(text)
    (dest/'README.md').write_text('# Minimal source-only VLA experiment\n\n**B task success was not achieved: 0/2 before and 0/2 after adaptation.** Mean XY goal error decreased from 20.47 cm to 10.95 cm. Source A succeeded 1/2.\n\n[Full measured report](FINAL_REPORT.md) · [English PDF](SOURCE_ONLY_VLA_PILOT_REPORT.pdf) · [Video comparison: scene 45](videos/comparison_demo_45.mp4) · [Video comparison: scene 46](videos/comparison_demo_46.mp4) · [Offline processing video](videos/offline_source_processing.mp4).\n\nOnly existing A records and offline B kinematics/static rendering supplied training data. No executed B training demonstrations. Original VLA weights were retained; 334,592 LoRA parameters received 100 updates using 189 processed anchors from three source episodes.\n')
    files=[]
    for p in sorted(dest.rglob('*')):
        if p.is_file() and p.name!='artifact_manifest.json':files.append(dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (dest/'artifact_manifest.json').write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),files=files),indent=2))
    print(dest)


if __name__=='__main__':main()
