#!/usr/bin/env python3
"""Verify a completed duration experiment and export English loss/PDF evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];raw=Path(args.run_dir).resolve();out=root/'results'/raw.name
    train=json.loads((raw/'training.json').read_text());comparison=json.loads((raw/'comparison.json').read_text())
    parent=Path(train['parent_run']);rows=json.loads((raw/'evaluation.json').read_text())
    assert 600<=train['additional_training_seconds']<610
    assert train['additional_steps']>0 and train['steps']==train['parent_steps']+train['additional_steps']
    assert train['parent_adapter_sha256']==hashlib.sha256((parent/'adapter.safetensors').read_bytes()).hexdigest()
    assert train['processed_training_data_sha256']==hashlib.sha256((parent/'processed_source_data.npz').read_bytes()).hexdigest()
    assert train['adapter_sha256']==hashlib.sha256((raw/'adapter.safetensors').read_bytes()).hexdigest()
    assert train['adapter_sha256']!=train['parent_adapter_sha256']
    assert train['executed_target_training_episodes']==0 and train['training_frames']==189
    assert len(rows)==4 and all(r['training_usage']=='none_final_evaluation_only' for r in rows)
    old=json.loads((parent/'evaluation.json').read_text())
    max_delta=0.
    for r in rows:
        if r['condition']=='B_previous100_UR5e':
            baseline=next(x for x in old if x['condition']=='B_adapted_UR5e' and x['source_initial_scene_id']==r['source_initial_scene_id'])
            assert r['success']==baseline['success']
            delta=abs(r['goal_error_m']-baseline['goal_error_m']);max_delta=max(max_delta,delta)
            assert delta<1e-5
    with (raw/'adapter.safetensors').open('rb') as f:
        header=json.loads(f.read(int.from_bytes(f.read(8),'little')))
    keys=[k for k in header if k!='__metadata__'];assert all('lora_' in k for k in keys)
    count=0
    for k in keys:
        n=1
        for dim in header[k]['shape']:n*=dim
        count+=n
    assert count==334592
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    log=json.loads((raw/'train_log.json').read_text())
    fig,ax=plt.subplots(figsize=(8.4,3.7),layout='constrained')
    ax.plot([r['additional_step'] for r in log],[r['mean_recent_loss'] for r in log],color='#1678a5',linewidth=2)
    ax.set(xlabel='Additional updates after the previous 100-update adapter',ylabel='Mean recent training loss',
           title='Same source-derived data: ten-minute LoRA continuation')
    ax.set_yscale('log');ax.grid(alpha=.2)
    fig.savefig(out/'training_loss.png',dpi=160);plt.close(fig)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph,Table,TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    path=out/'TEN_MINUTE_TRAINING_REPORT.pdf';w,h=A4
    pdf=canvas.Canvas(str(path),pagesize=A4,initialFontName='DejaVu',initialFontSize=10)
    style=ParagraphStyle('body',fontName='DejaVu',fontSize=10.5,leading=16)
    y=h-55
    def paragraph(text):
        nonlocal y
        p=Paragraph(text,style);_,height=p.wrap(w-84,h)
        if y-height<45:pdf.showPage();y=h-55
        p.drawOn(pdf,42,y-height);y-=height+14
    pdf.setFont('DejaVuBold',17);pdf.drawString(42,y,'Ten-minute B VLA adapter continuation');y-=35
    paragraph('Task: push the plate to the front of the stove. Robot B is UR5e. All training data are the same 189 offline-rendered anchors derived from three existing Panda demonstrations; executed B training demonstrations remain zero.')
    paragraph(f"The previous adapter received 100 updates. Starting from its exact weights, the extension ran for <b>{train['additional_training_seconds']:.2f} seconds</b>, adding <b>{train['additional_steps']:,} updates</b> for <b>{train['steps']:,} total updates</b>. Batch four, LR 1e-4 and rank-eight LoRA (334,592 trainable parameters) remain unchanged.")
    data=[['B policy','Success / trials','Mean XY error'],
          ['Previous 100 updates',f"{comparison['previous_successes']}/2",f"{comparison['previous_mean_goal_error_m']*100:.2f} cm"],
          ['Plus ten minutes',f"{comparison['extended_successes']}/2",f"{comparison['extended_mean_goal_error_m']*100:.2f} cm"]]
    table=Table(data,colWidths=[247,112,112]);table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DejaVu'),('FONTSIZE',(0,0),(-1,-1),10),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9f4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#8da3b9')),('BOTTOMPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),10)]));_,height=table.wrap(w-84,h);table.drawOn(pdf,42,y-height);y-=height+20
    if comparison['extended_successes']>comparison['previous_successes']:
        paragraph('More training improved observed task success on these two repeated scenes. This supports training duration as one contributing factor in this pilot, but does not establish a reliable success rate or eliminate data-processing limitations.')
    else:
        paragraph('More training did not increase observed task success on these two repeated scenes. A lower training loss alone cannot establish that insufficient updates caused the earlier task failures.')
    if comparison['extended_mean_goal_error_m']>comparison['previous_mean_goal_error_m']:
        paragraph('Mean final goal error became worse after the extension. This is consistent with a possible generalization or source-processing mismatch, but no validation-loss or contact ablation was run to identify the cause.')
    paragraph(f"The old adapter was re-evaluated under identical settings; its maximum final goal-error difference from the original run was {max_delta:.3g} m. Both conditions execute five predicted actions before replanning, with ten flow steps per query and a maximum of 200 simulator steps.")
    paragraph('The original pilot saved no optimizer state. We therefore retained its adapter weights but initialized AdamW afresh. The extension saves optimizer/RNG state for future exact resumes. This is weight continuation, not an exact optimizer/RNG continuation.')
    pdf.showPage();y=h-55;pdf.setFont('DejaVuBold',17);pdf.drawString(42,y,'Training evidence and practical limits');y-=35
    pdf.drawImage(str(out/'training_loss.png'),42,y-225,width=w-84,height=225);y-=245
    paragraph('The curve summarizes recent training-batch loss, not validation loss or task success. Sampling is uniform with replacement, so the update count is more precise than an epoch count for this loop.')
    paragraph('The checkpoint is the last time-budget checkpoint. No target rollout, target action label or target reward is used for training or checkpoint selection. The old and new policies use the same two already-examined test scenes; this is a repeated-scene duration comparison, not an independent generalization benchmark.')
    paragraph('Source processing still copies object motion from A and calculates hypothetical B poses without target contact dynamics. Increasing training duration cannot by itself verify friction, pushing force or contact consistency in that synthetic data.')
    paragraph('Review the two English training_duration_comparison movies beside FINAL_REPORT.md. All four final B rollout movies, hashes, training logs and measured results are included.')
    pdf.save()
    import fitz
    doc=fitz.open(path);pdf_checks=[]
    qa=out/'qa';qa.mkdir(exist_ok=True)
    for i,page in enumerate(doc):
        text=page.get_text();assert '\ufffd' not in text and '\u25a0' not in text
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox'];assert min(x0,y0)>=0 and x1<=page.rect.width+.5 and y1<=page.rect.height+.5
        page.get_pixmap(matrix=fitz.Matrix(1.1,1.1)).save(qa/f'page_{i+1:02}.png');pdf_checks.append({'page':i+1,'layout_passed':True})
    videos=[]
    for p in (out/'videos').glob('*.mp4'):
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(p)],text=True));assert float(probe['format']['duration'])>0
        videos.append(p.name)
    result=dict(status='passed',checks=['600_seconds_completed','exact_parent_adapter_weights','unchanged_source_training_data',
                'new_adapter_hash_changed','zero_executed_B_training_demonstrations','only_LoRA_weights_saved',
                'previous_policy_reproduced','four_matched_B_evaluations','all_videos_decode','PDF_layout_and_text'],
                previous_policy_max_goal_error_delta_m=max_delta,videos=videos,pdf_pages=pdf_checks)
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    for name in ['README.md','FINAL_REPORT.md']:
        p=out/name
        base=p.read_text().split('\n\n[English PDF]')[0].split('\n\n## Additional report artifacts')[0]
        p.write_text(base+'\n\n## Additional report artifacts\n\n[English PDF](TEN_MINUTE_TRAINING_REPORT.pdf) · [Training loss](training_loss.png) · [Verification](verification.json).\n\nReproduce with `bash scripts/launch_vla_extra_training.sh` after the original pilot materials are installed. The tmux job performs 600 seconds of additional training, fixed evaluation and report export independently of SSH.\n')
    (out/'artifact_manifest.json').write_text(json.dumps({'files':[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json']},indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
