#!/usr/bin/env python3
"""Build and audit the editable English technical report; no experiment reruns.

Usage: .venv/bin/python scripts/build_english_report.py
The JSON source and original experiment evidence are independent of report layout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.parse
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
                               PageBreak, Table, TableStyle, Flowable, KeepTogether, Image)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/english_technical_report_20261106'
QA=OUT/'qa'
PDF_NAME='Cross_Robot_Real2Sim_Technical_Report.pdf'
INK=colors.HexColor('#14283F'); BLUE=colors.HexColor('#12617A'); TEAL=colors.HexColor('#0F7E85')
MUTED=colors.HexColor('#5A6C7E'); LIGHT=colors.HexColor('#EAF3F6'); BORDER=colors.HexColor('#D3E0E7')
PAGE_W,PAGE_H=A4
MARGIN=47
BODY_W=PAGE_W-2*MARGIN
FONT_DIR=Path('/usr/share/fonts/truetype/dejavu')


def register_fonts():
    for name,file in [('DejaVu','DejaVuSans.ttf'),('DejaVu-Bold','DejaVuSans-Bold.ttf'),
                      ('DejaVu-Italic','DejaVuSans-Oblique.ttf'),('DejaVu-Mono','DejaVuSansMono.ttf')]:
        path=FONT_DIR/file
        if not path.is_file():raise FileNotFoundError(f'Required embeddable font absent: {path}')
        pdfmetrics.registerFont(TTFont(name,str(path)))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu-Italic',boldItalic='DejaVu-Bold')


def styles():
    return {
      'body':ParagraphStyle('Body',fontName='DejaVu',fontSize=9.6,leading=14.0,textColor=INK,spaceAfter=8.3,splitLongWords=True),
      'small':ParagraphStyle('Small',fontName='DejaVu',fontSize=8.1,leading=11.5,textColor=MUTED,spaceAfter=6),
      'h':ParagraphStyle('H',fontName='DejaVu-Bold',fontSize=11.4,leading=15,textColor=BLUE,spaceBefore=10,spaceAfter=6,keepWithNext=True),
      'title':ParagraphStyle('Title',fontName='DejaVu-Bold',fontSize=20,leading=26,textColor=INK,spaceAfter=11),
      'cover':ParagraphStyle('Cover',fontName='DejaVu-Bold',fontSize=30,leading=37,textColor=INK,spaceAfter=16),
      'eyebrow':ParagraphStyle('Eyebrow',fontName='DejaVu-Bold',fontSize=8.5,leading=12,textColor=TEAL,spaceAfter=8),
      'subtitle':ParagraphStyle('Subtitle',fontName='DejaVu',fontSize=10.3,leading=15.4,textColor=MUTED,spaceAfter=14),
      'cell':ParagraphStyle('Cell',fontName='DejaVu',fontSize=8.2,leading=11.5,textColor=INK,spaceAfter=0,splitLongWords=True),
      'headcell':ParagraphStyle('HeadCell',fontName='DejaVu-Bold',fontSize=8.0,leading=11.2,textColor=colors.white,spaceAfter=0,splitLongWords=True),
      'code':ParagraphStyle('Code',fontName='DejaVu-Mono',fontSize=7.6,leading=11,textColor=INK,spaceAfter=0,splitLongWords=True),
    }


def markup(text):
    # JSON prose may use safe ReportLab markup; links are generated from local paths.
    return str(text)


def link(path,label=None):
    label=label or path
    target='https://github.com/Doyoung619/Cross_Robot_Real2Sim/'+('tree/main/' if path.endswith('/') else 'blob/main/')+urllib.parse.quote(path,safe='/')
    return f'<link href="{escape(target)}" color="#12617A"><u>{escape(label)}</u></link>'


class Graphic(Flowable):
    def __init__(self,name,data=None,height=130):
        super().__init__();self.name=name;self.data=data or {};self.width=BODY_W;self.height=height
    def draw(self):
        c=self.canv
        if self.name=='pipeline':
            labels=[('1','Selection','Panda HDF5 metadata'),('2','Task','Provided TaskSpec'),('3','Retarget','Executed UR5e teachers'),('4','Learn','Saved MLP / held-out eval'),('5','Diagnose','Rules / bounded iteration')]
            w=(self.width-16)/5
            for i,(n,title,detail) in enumerate(labels):
                x=i*w
                c.setFillColor(LIGHT if i%2==0 else colors.HexColor('#F3F7FA'));c.roundRect(x,12,w-5,self.height-22,6,fill=1,stroke=0)
                c.setFillColor(TEAL);c.setFont('DejaVu-Bold',17);c.drawString(x+9,self.height-36,n)
                c.setFillColor(INK);c.setFont('DejaVu-Bold',8.8);c.drawString(x+9,self.height-53,title)
                p=Paragraph(detail,ParagraphStyle('Diagram',fontName='DejaVu',fontSize=7.3,leading=10,textColor=MUTED))
                _,h=p.wrap(w-20,55);p.drawOn(c,x+9,self.height-63-h)
                if i<4:
                    c.setStrokeColor(TEAL);c.line(x+w-3,55,x+w+3,55)
            c.setFillColor(MUTED);c.setFont('DejaVu',7.5);c.drawString(0,0,'Solid pipeline: executed bootstrap/fast. Human-video models remain an untested full-mode branch.')
        elif self.name=='success':
            items=[('Bootstrap / before',0,3),('Bootstrap / after',1,3),('Fast / before',9,10),('Fast / after',9,10)]
            for i,(label,success,total) in enumerate(items):
                y=self.height-27-i*30
                c.setFont('DejaVu',8.3);c.setFillColor(INK);c.drawString(0,y,label)
                x=140;w=self.width-205
                c.setFillColor(LIGHT);c.roundRect(x,y-3,w,13,3,fill=1,stroke=0)
                if success:
                    c.setFillColor(TEAL if 'after' in label else BLUE);c.roundRect(x,y-3,w*success/total,13,3,fill=1,stroke=0)
                c.setFillColor(INK);c.drawString(x+w+10,y,f'{success}/{total}')
        elif self.name=='lineage':
            items=[('Source demos','3, 5, 7, 4, 1'),('UR5e teachers','10 executed successes'),('Shared B model','One baseline checkpoint'),('Held-out cases','10001 / 10003 / 10009')]
            w=(self.width-30)/4
            for i,(title,desc) in enumerate(items):
                x=i*(w+10)
                c.setFillColor(LIGHT);c.roundRect(x,17,w,self.height-30,5,fill=1,stroke=0)
                p=Paragraph(f'<b>{title}</b><br/>{desc}',ParagraphStyle('Lin',fontName='DejaVu',fontSize=8.2,leading=12,textColor=INK))
                _,h=p.wrap(w-14,100);p.drawOn(c,x+7,self.height-24-h)
                if i<3:
                    c.setStrokeColor(TEAL);c.line(x+w+1,self.height/2,x+w+9,self.height/2)
        elif self.name=='action':
            boxes=[('26D state','Training-only scaling'),('MLP 26-128-128-4','XYZ + gripper logit'),('4D intermediary','Clip XYZ; threshold grip'),('7D OSC vector','XYZ / zero rot / grip')]
            w=(self.width-30)/4
            for i,(title,desc) in enumerate(boxes):
                x=i*(w+10);c.setFillColor(LIGHT);c.roundRect(x,13,w,self.height-24,5,fill=1,stroke=0)
                p=Paragraph(f'<b>{title}</b><br/>{desc}',ParagraphStyle('Act',fontName='DejaVu',fontSize=7.5,leading=11,textColor=INK))
                _,h=p.wrap(w-14,80);p.drawOn(c,x+7,self.height-23-h)
                if i<3:c.setStrokeColor(TEAL);c.line(x+w+1,self.height/2,x+w+9,self.height/2)


class ReportDoc(BaseDocTemplate):
    def __init__(self,path,total):
        super().__init__(str(path),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,
                         topMargin=75,bottomMargin=49,title='Cross-Robot Real2Sim - Technical Report',
                         author='Cross-Robot Real2Sim project',pageCompression=1)
        self.total=total;self.layout=[]
        frame=Frame(MARGIN,49,BODY_W,PAGE_H-124,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate('Report',frames=[frame],onPage=self.page_chrome))
    def page_chrome(self,canvas,doc):
        canvas.saveState();canvas.setFillColor(MUTED);canvas.setFont('DejaVu-Bold',7.1)
        canvas.drawString(MARGIN,PAGE_H-33,'CROSS-ROBOT REAL2SIM  /  TECHNICAL REPORT')
        canvas.setFont('DejaVu',7.1);canvas.drawRightString(PAGE_W-MARGIN,PAGE_H-33,'06 NOV 2026')
        canvas.setStrokeColor(BORDER);canvas.line(MARGIN,PAGE_H-43,PAGE_W-MARGIN,PAGE_H-43)
        canvas.line(MARGIN,39,PAGE_W-MARGIN,39)
        canvas.setFont('DejaVu',7);canvas.drawString(MARGIN,25,'Simulation evidence / oracle_sim_state / English edition')
        canvas.drawRightString(PAGE_W-MARGIN,25,f'{doc.page:02d} / {self.total:02d}')
        canvas.restoreState()
    def afterFlowable(self,f):
        if isinstance(f,(Paragraph,Table,Graphic)):
            self.layout.append({'page':self.page,'kind':type(f).__name__,'bottom_y':round(self.frame._y,3),'height':round(getattr(f,'height',getattr(f,'_height',0)),3)})


def make_table(block,sty):
    rows=[]
    if block.get('headers'):rows.append([Paragraph(markup(x),sty['headcell']) for x in block['headers']])
    rows.extend([[Paragraph(markup(x),sty['cell']) for x in row] for row in block['rows']])
    weights=block.get('widths',[1]*len(rows[0]));total=sum(weights)
    table=Table(rows,colWidths=[BODY_W*x/total for x in weights],hAlign='LEFT',repeatRows=1 if block.get('headers') else 0)
    commands=[('FONTNAME',(0,0),(-1,-1),'DejaVu'),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,BORDER)]
    start=0
    if block.get('headers'):
        commands.append(('BACKGROUND',(0,0),(-1,0),BLUE));start=1
    for row in range(start,len(rows)):
        commands.append(('BACKGROUND',(0,row),(-1,row),colors.HexColor('#F1F6F8') if row%2 else colors.white))
        commands.append(('LINEBELOW',(0,row),(-1,row),.35,BORDER))
    table.setStyle(TableStyle(commands));return table


def source_validation_block():
    folder=ROOT/'results/english_case_studies_20261106/source'
    rows=[]
    for demo in [3,5,7]:
        p=folder/f'source_demo_{demo}.json'
        if not p.exists():continue
        data=json.loads(p.read_text())
        rows.append([f'demo_{demo}',f"{data['validated_frames']} frames<br/>{data['video_duration_s']:.2f} s at {data['video_fps']:.0f} Hz",
                     f"Can: {data['max_can_position_error_m']:.1f}<br/>EEF: {data['max_eef_position_error_m']:.2e}<br/>Tolerance: {data['tolerance_m']:.0e}",link(str(p.relative_to(ROOT)),'All-frame source metadata')])
    return {'type':'table','headers':['Source','Recorded duration','Max position error (m)','Evidence'],'rows':rows,'widths':[.8,1.2,1.7,1.6]} if rows else None


def filmstrip(block,sty):
    import imageio.v2 as imageio
    from PIL import Image as PILImage
    from imageio_ffmpeg import count_frames_and_secs
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    image_row=[];caption_row=[];records=[]
    width=(BODY_W-16)/3
    for i,panel in enumerate(block['panels']):
        video=ROOT/panel['path']
        count,duration=count_frames_and_secs(str(video));index=min(count-1,int(count*float(panel.get('fraction',.65))))
        reader=imageio.get_reader(str(video))
        try:frame=reader.get_data(index)
        finally:reader.close()
        path=folder/f'case01_panel_{i+1:02d}.png';PILImage.fromarray(frame).save(path)
        h=width*frame.shape[0]/frame.shape[1]
        image_row.append(Image(str(path),width=width,height=h))
        caption_row.append(Paragraph(panel['label'],sty['cell']))
        records.append({'video':panel['path'],'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'frame_index':index,'frame_count':count,'video_duration_s':duration,'figure':str(path.relative_to(ROOT))})
    (folder/'frame_provenance.json').write_text(json.dumps(records,indent=2)+'\n')
    table=Table([image_row,caption_row],colWidths=[BODY_W/3]*3)
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DejaVu'),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return table


def audit_pdf(pdf,source,layout):
    import fitz
    from pypdf import PdfReader
    document=fitz.open(pdf);reader=PdfReader(str(pdf))
    page_info=[];issues=[];fonts=[]
    for i,page in enumerate(document):
        text=page.get_text()
        if re.search('[\uac00-\ud7af\u3400-\u9fff]',text):issues.append(f'Non-English CJK text on page {i+1}')
        if '\ufffd' in text:issues.append(f'Replacement glyph on page {i+1}')
        if len(text.strip())<150:issues.append(f'Unexpected low text content on page {i+1}')
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        for span in spans:
            x0,y0,x1,y1=span['bbox']
            if x0<43 or x1>PAGE_W-43 or y0<15 or y1>PAGE_H-15:
                issues.append(f'Out-of-margin text on page {i+1}: {span["text"][:45]}')
        page_info.append({'page':i+1,'characters':len(text),'links':len(page.get_links()),'text_bounds':[round(min(s['bbox'][0] for s in spans),2),round(min(s['bbox'][1] for s in spans),2),round(max(s['bbox'][2] for s in spans),2),round(max(s['bbox'][3] for s in spans),2)]})
    for page in reader.pages:
        for font in page['/Resources'].get('/Font',{}).values():
            obj=font.get_object();desc=obj.get('/FontDescriptor')
            descriptor=desc.get_object() if desc else {}
            embedded=any(key in descriptor for key in ['/FontFile','/FontFile2','/FontFile3'])
            fonts.append({'name':str(obj.get('/BaseFont')),'embedded':embedded})
            if not embedded:issues.append(f'Unembedded font {obj.get("/BaseFont")}')
    if len(document)!=len(source['pages']):issues.append(f'Unexpected page overflow: {len(document)} vs intended {len(source["pages"])}')
    if any(item['bottom_y']<48 for item in layout):issues.append('Flowable crossed the body bottom margin')
    qa={'status':'passed' if not issues else 'failed','page_count':len(document),'intended_page_count':len(source['pages']),
        'english_text_check':'passed' if not any('English' in x for x in issues) else 'failed',
        'embedded_fonts':list({x['name']:x for x in fonts}.values()),'pages':page_info,'issues':issues,
        'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'validation_tools':['ReportLab layout audit','PyMuPDF text bounds','pypdf font descriptors','pdftotext'],'body_bottom_margin_points':49}
    for offset in range(0,len(document),6):
        from PIL import Image,ImageDraw
        thumbs=[]
        for page in list(document)[offset:offset+6]:
            pix=page.get_pixmap(matrix=fitz.Matrix(.60,.60),alpha=False)
            thumbs.append(Image.frombytes('RGB',[pix.width,pix.height],pix.samples))
        thumb_w=max(x.width for x in thumbs);thumb_h=max(x.height for x in thumbs)
        sheet=Image.new('RGB',(thumb_w*2+36,thumb_h*3+65),'#dce4e9');draw=ImageDraw.Draw(sheet)
        for j,img in enumerate(thumbs):
            x=12+(j%2)*(thumb_w+12);y=20+(j//2)*(thumb_h+16)
            sheet.paste(img,(x,y));draw.text((x,y-13),f'Page {offset+j+1}',fill='black')
        sheet.save(QA/f'contact_sheet_{offset//6+1:02d}.png')
    for number in (4,7,10,15,16,17):
        if number<=len(document):document[number-1].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(QA/f'qa_page_{number:02d}.png')
    (QA/'qa_report.json').write_text(json.dumps(qa,indent=2)+'\n')
    (QA/'layout_audit.json').write_text(json.dumps(layout,indent=2)+'\n')
    subprocess.run(['pdftotext','-layout',str(pdf),str(QA/'extracted_text.txt')],check=True)
    return qa


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=OUT/'report_source.json');parser.add_argument('--output',type=Path,default=OUT/PDF_NAME)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.source.read_text());register_fonts();sty=styles();story=[]
    for number,page in enumerate(source['pages']):
        if number:story.append(PageBreak())
        story.append(Paragraph(markup(page.get('eyebrow',f'TECHNICAL DETAIL {number+1:02d}')),sty['eyebrow']))
        story.append(Paragraph(markup(page['title']),sty['cover'] if number==0 else sty['title']))
        if page.get('subtitle'):story.append(Paragraph(markup(page['subtitle']),sty['subtitle']))
        for block in page['blocks']:
            kind=block['type']
            if kind in ('p','small','h'):
                story.append(Paragraph(markup(block['text']),sty['body' if kind=='p' else kind]))
            elif kind=='table':story.extend([make_table(block,sty),Spacer(1,9)])
            elif kind=='callout':
                inner=Paragraph(f'<b>{block["title"]}</b><br/>{block["text"]}',sty['body'])
                table=Table([[inner]],colWidths=[BODY_W]);table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DejaVu'),('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.7,BORDER),('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story.extend([table,Spacer(1,10)])
            elif kind=='code':
                content='<br/>'.join(escape(x).replace(' ','&nbsp;') for x in block['text'].splitlines())
                table=Table([[Paragraph(content,sty['code'])]],colWidths=[BODY_W]);table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DejaVu'),('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#F1F4F7')),('LEFTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]));story.extend([table,Spacer(1,9)])
            elif kind=='graphic':story.extend([Graphic(block['name'],block.get('data'),block.get('height',130)),Spacer(1,9)])
            elif kind=='filmstrip':story.extend([filmstrip(block,sty),Spacer(1,6)])
            elif kind=='source_validation':
                dynamic=source_validation_block()
                if dynamic:story.extend([make_table(dynamic,sty),Spacer(1,7)])
                else:story.append(Paragraph('Source replay metadata are linked in the case-study index; no unavailable per-frame validation measurement is invented in this report.',sty['body']))
            elif kind=='links':
                for item in block['items']:story.append(Paragraph(link(item['path'],item.get('label')),sty['small']))
            else:raise ValueError(f'Unknown report block {kind}')
    doc=ReportDoc(args.output,len(source['pages']));doc.build(story,canvasmaker=lambda *a,**kw:Canvas(*a,**{**kw,'initialFontName':'DejaVu'}))
    qa=audit_pdf(args.output,source,doc.layout)
    print(json.dumps({'pdf':str(args.output),'page_count':qa['page_count'],'qa':qa['status'],'issues':qa['issues']},indent=2))
    inputs=[args.source,Path(__file__),ROOT/'requirements-report.txt',ROOT/'results/english_case_studies_20261106/case_manifest.json',ROOT/'results/review_20261105/problem_evidence.json']
    inputs.extend((ROOT/'results/english_case_studies_20261106/source').glob('source_demo_*.json'))
    for run_id in source['source_run_ids']:
        for name in ('01_selection/selection.json','01_selection/source_inspection.json','02_task/task_spec.json','03_retarget/attempts.csv','04_policy/dataset_metadata.json','04_policy/eval.csv','04_policy/train_metrics.json','05_diagnosis/before_after.json','environment.json','config.yaml'):
            inputs.append(ROOT/'results'/run_id/name)
    manifest={'schema_version':'1.0','language':'en','page_count':qa['page_count'],'pdf':str(args.output.relative_to(ROOT)),'pdf_sha256':qa['pdf_sha256'],'report_generation_runs_training':False,'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs if p.is_file()],'qa_report':'qa/qa_report.json'}
    (OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if qa['issues']:raise SystemExit(1)


if __name__=='__main__':main()
