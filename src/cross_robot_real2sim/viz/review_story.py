"""Readable five-stage movies made from real metadata, logs and frozen replays."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .review_rollouts import font
from ..common.provenance import file_sha256

FAST="fast_20261105T182127Z"
BOOT="bootstrap_20261105T181834Z"
SIZE=(1280,720)
BG=(13,21,36)


def wrap(draw,text,typeface,width):
    lines=[];line=""
    for char in text:
        if line and draw.textlength(line+char,font=typeface)>width:
            lines.append(line);line=""
        line+=char
    if line:lines.append(line)
    return lines


class Movie:
    def __init__(self,path):
        import imageio.v2 as imageio
        self.path=path
        self.writer=imageio.get_writer(str(path),fps=10,codec="libx264",quality=7)
        self.frames=0
        self.chapters=[]

    def write(self,im):
        self.writer.append_data(np.asarray(im))
        self.frames+=1

    def heading(self,title):
        im=Image.new("RGB",SIZE,BG)
        draw=ImageDraw.Draw(im)
        draw.text((24,17),title,font=font(31),fill="white")
        draw.text((25,62),"사람 영상 없는 경로 · oracle_sim_state · 저장 결과 및 동일 모델 재실행",font=font(20),fill=(255,207,111))
        return im

    def chapter(self,title,start,sources):
        self.chapters.append({"title":title,"start_seconds":start/10,"end_seconds":self.frames/10,"source_ids":sources})

    def card(self,title,lines,seconds=4,sources=()):
        start=self.frames
        im=self.heading(title);draw=ImageDraw.Draw(im)
        y=124
        for text in lines:
            for line in wrap(draw,text,font(27),1200):
                draw.text((30,y),line,font=font(27),fill=(224,234,248));y+=42
            y+=13
        if y>700:raise ValueError(f"Card text overflow: {title}")
        for _ in range(round(seconds*10)):self.write(im)
        self.chapter(title,start,list(sources))

    def scene(self,title,panels,notes=(),seconds=None):
        import imageio.v2 as imageio
        start=self.frames
        opened=[]
        try:
            for label,path in panels:
                path=Path(path)
                if path.suffix.lower() in (".png",".jpg"):
                    opened.append((label,path,None,1,Image.open(path).convert("RGB")))
                else:
                    reader=imageio.get_reader(path)
                    opened.append((label,path,reader,reader.count_frames(),None))
            count=round(seconds*10) if seconds else max(n for _,_,reader,n,_ in opened if reader is not None)
            for index in range(count):
                im=self.heading(title);draw=ImageDraw.Draw(im)
                panel_width=1240//len(opened)
                for slot,(label,path,reader,length,still) in enumerate(opened):
                    frame=Image.fromarray(reader.get_data(min(index,length-1))) if reader is not None else still.copy()
                    frame.thumbnail((panel_width-12,498),Image.Resampling.LANCZOS)
                    x=20+slot*panel_width+(panel_width-frame.width)//2
                    y=123+(498-frame.height)//2
                    im.paste(frame,(x,y))
                    draw.text((24+slot*panel_width,95),label,font=font(19),fill="white")
                    if reader is not None and index>=length:
                        draw.rectangle((x,y+frame.height-24,x+frame.width,y+frame.height),fill="black")
                        draw.text((x+5,y+frame.height-24),"해당 실행 종료: 마지막 프레임 유지",font=font(15),fill=(255,220,130))
                y=636
                for note in notes:
                    draw.text((25,y),note,font=font(19),fill=(215,228,244));y+=27
                self.write(im)
            self.chapter(title,start,[str(path) for _,path,_,_,_ in opened])
        finally:
            for _,_,reader,_,_ in opened:
                if reader is not None:reader.close()

    def close(self):
        self.writer.close()
        return {"path":str(self.path),"frames":self.frames,"fps":10,"duration_seconds":self.frames/10,"chapters":self.chapters}


def plot_evidence(root,output,run_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run=root/"runs"/run_id
    plots=output/"figures";plots.mkdir(exist_ok=True)
    mode=run_id.split("_")[0]
    with np.load(run/"03_retarget/source_trace.npz",allow_pickle=False) as data:
        meta=json.loads(str(data["metadata_json"].item()))
        t=data["timestamps"];relative=data["relative_eef"]
        grasp=int(data["grasp_index"])
        fig,axes=plt.subplots(1,2,figsize=(12,4),dpi=140)
        for axis,label in enumerate("xyz"):
            axes[0].plot(t,relative[:,axis],label=f"EEF - Can {label}")
        axes[0].axvline(t[grasp],color="black",ls="--",label="extracted grasp")
        axes[0].set(xlabel="Reconstructed source time (s)",ylabel="World-aligned relative position (m)",title=f"Measured Panda {meta['demo_id']}")
        axes[0].legend(fontsize=8)
        axes[1].plot(t,data["object_pos"][:,2]-data["object_pos"][0,2])
        axes[1].set(xlabel="Reconstructed source time (s)",ylabel="Can lift above initial position (m)",title="Source lift geometry used by teacher")
        fig.tight_layout();fig.savefig(plots/f"{mode}_source_geometry.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),dpi=140)
    for label,folder in [("before","04_policy"),("after","05_diagnosis/feedback_run/04_policy")]:
        rows=list(csv.DictReader((run/folder/"train_log.csv").open()))
        axes[0].plot([int(r["step"]) for r in rows],[float(r["xyz_mse"]) for r in rows],label=label)
        axes[1].plot([int(r["step"]) for r in rows],[float(r["gripper_bce"]) for r in rows],label=label)
    for axis in axes:axis.set_xlabel("Actual optimizer step");axis.legend()
    axes[0].set_ylabel("XYZ MSE");axes[0].set_yscale("log")
    axes[1].set_ylabel("Gripper BCE")
    fig.suptitle("Recorded training loss (not rollout success)")
    fig.tight_layout();fig.savefig(plots/f"{mode}_training_loss.png");plt.close(fig)


def verify_media(output):
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    records=[]
    for path in sorted(output.rglob("*.mp4")):
        count,duration=count_frames_and_secs(str(path))
        if count<=0 or duration<=0:raise ValueError(f"Empty video: {path}")
        reader=imageio.get_reader(path)
        try:
            first,last=reader.get_data(0),reader.get_data(count-1)
            if first.shape!=last.shape or first.shape[-1]!=3:
                raise ValueError(f"Invalid frames: {path}")
        finally:reader.close()
        records.append({"path":str(path.relative_to(output)),"frames":count,"duration_seconds":duration,
                        "frame_shape":list(first.shape),"bytes":path.stat().st_size,"sha256":file_sha256(path)})
    return records


def assemble(root:Path,output:Path,raw:Path,cases:list[dict]):
    videos=output/"videos";videos.mkdir(exist_ok=True)
    by_id={case["case_id"]:case for case in cases}
    def clip(case_id):return output/by_id[case_id]["video_path"]
    for run_id in (FAST,BOOT):plot_evidence(root,output,run_id)
    movies=[]
    for run_id,name in [(FAST,"01_pipeline_fast.mp4"),(BOOT,"02_pipeline_bootstrap_feedback.mp4")]:
        run=root/"runs"/run_id
        mode=run_id.split("_")[0]
        selection=json.loads((run/"01_selection/selection.json").read_text())
        task=json.loads((run/"02_task/task_spec.json").read_text())
        stats=json.loads((run/"03_retarget/stats.json").read_text())
        dataset=json.loads((run/"04_policy/dataset_metadata.json").read_text())
        correction=json.loads((run/"05_diagnosis/before_after.json").read_text())
        before,after=correction["before"],correction["after"]
        movie=Movie(videos/name)
        try:
            movie.card(f"5단계 실행 증거 | {mode}",[
                f"원본 실행: {run_id}",
                "영상 입력: 없음. 1·2단계는 메타데이터와 제공된 작업 명세 사용.",
                "로봇 영상은 저장된 제어기·체크포인트의 실제 시뮬레이터 재실행.",
                "새로운 학습 없음. 성공·실패 및 피드백 효과를 구분해서 표시."],seconds=5)
            movie.card("1단계 | 실제 Panda 데모 선택",[
                "입력: 공개 robomimic Can / PH / v1.5 HDF5",
                f"선택된 데모: {', '.join(selection['selected_source_demonstration_ids'])}",
                f"선택 전략: {selection['selection_strategy']}",
                f"사람 영상 선택 상태: {selection['human_clip_selection']} (CLIP 미실행)"],
                sources=[str(run/"01_selection/selection.json")])
            movie.card("2단계 | 목표와 제약 정의",[
                f"작업 출처: {task['task_source']} — 직접 제공한 작업 명세",
                f"목표: {task['object']} → {task['goal']['relation']} → {task['goal']['receptacle']}",
                "순서: 접근 → 잡기 → 들기 → 운반 → 내리기 → 놓기 → 물러나기",
                "질량·마찰·접촉 추론: unknown. 목표 좌표는 시뮬레이터 제공.",
                "사람 영상 해석(Qwen)은 실행하지 않음."],sources=[str(run/"02_task/task_spec.json")])
            movie.scene("3단계 | 실제 원본 데모에서 추출한 정보",[("HDF5 측정값: 잡는 상대 위치와 들어 올린 높이",output/"figures"/f"{mode}_source_geometry.png")],
                        notes=["전체 관절 궤적 변환이 아닌, 잡기·들기 기하정보를 규칙 기반 제어에 사용.","이 그림은 실제 숫자 데이터이며 사람 영상이나 로봇 재생 영상이 아님."],seconds=4)
            panels=[("UR5e 교사 실행",clip(f"{mode}_teacher_1042"))]
            if mode=="fast":panels.insert(0,("Panda 규칙 정책: 별도 실행 분기",clip("fast_panda_scripted_42")))
            movie.scene("3단계 | 실행된 로봇 시범: 목표 칸 표시",panels,
                        notes=[f"기록된 원본 정보 기반 UR5e 성공: {stats['counts']['source_retargeted']['successes']}/{stats['counts']['source_retargeted']['attempts']}",
                               "초록색은 캔의 목표 칸. 종료 후에도 같은 제어기로 1초 추가 확인."])
            movie.scene("4단계 | 실제 학습 로그",[("기존 실행에서 기록된 손실",output/"figures"/f"{mode}_training_loss.png")],
                        notes=[f"학습 자료: 성공한 UR5e 교사 {len(dataset['episodes'])}개 / {dataset['samples']}개 상태-행동 쌍.",
                               "손실 감소와 실제 작업 성공률은 별도 지표."],seconds=4)
            if mode=="fast":
                movie.scene("4단계 | 학습된 신경망의 성공 사례",[("seed10001: 저장된 초기 정책",clip("fast_bc_before_10001")),("seed10009: 저장된 초기 정책",clip("fast_bc_before_10009"))],
                            notes=["영상이 실제로 성공한 사례인지 종료 상태와 1초 추가 검사를 함께 확인.","전체 원래 평가 성적: 9/10. 실패 시드10000도 별도 영상으로 보존."])
                movie.card("5단계 | 실패 분석과 한 번의 피드백",[
                    "초기 접근 실패: seed10000. 같은 조건에서 교사는 성공.",
                    f"교사 궤적 {correction['teacher_successes']}개 추가 후 재학습.",
                    f"동일한 평가 시드에서 성공: {before['successes']}/{before['episodes']} → {after['successes']}/{after['episodes']}",
                    f"전체 평균 XY 오차: {100*before['mean_placement_error']:.2f}cm → {100*after['mean_placement_error']:.2f}cm",
                    "이번 fast 피드백은 성공률 개선 없음."],seconds=5,sources=[str(run/"05_diagnosis/before_after.json")])
                movie.scene("5단계 | 같은 시드의 피드백 전·후",[("전: seed10001",clip("fast_bc_before_10001")),("후: seed10001",clip("fast_bc_after_10001"))],
                            notes=["둘 다 성공한 한 사례. 이 장면만으로 개선을 주장하지 않음.","집계 성적은 9/10 → 9/10이며, 전체 평균 오차는 조금 증가."])
            else:
                movie.scene("4단계 | 작은 데이터로 학습한 초기 정책의 실패",[("피드백 전 seed10002",clip("bootstrap_bc_before_10002"))],
                            notes=["원래 bootstrap 초기 평가: 0/3 성공.","접근 단계에서 멈춘 실패를 그대로 보존. 다음 장면과 초기 조건이 같음."])
                movie.card("5단계 | 추가 수집과 재학습",[
                    f"교사 궤적 {correction['teacher_successes']}개 추가; 동일한 평가 시드 유지.",
                    f"성공률: {before['successes']}/{before['episodes']} → {after['successes']}/{after['episodes']}",
                    "다음 비교는 seed10002 한 사례가 실패에서 성공으로 바뀐 실제 재실행.",
                    "다른 두 시드는 여전히 실패하므로 전체 성공으로 해석하지 않음."],seconds=5)
                movie.scene("5단계 | 실제로 실패 → 성공이 된 사례",[("전: seed10002",clip("bootstrap_bc_before_10002")),("후: 같은 seed10002",clip("bootstrap_bc_after_10002"))],
                            notes=["같은 시작 특징값 확인. 원본 체크포인트만 각각 로드해 재실행.","전체 집계: 0/3 → 1/3. 한 사례의 개선이며 일반화 증거는 부족."])
            movie.card("이 영상에서 확인한 범위",[
                "사람 영상 없는 5단계 경로의 데이터 연결과 실제 시뮬레이터 실행.",
                "제어기와 학습 정책을 구분한 성공 사례 및 피드백 반복.",
                "사람 영상 기반 full, 일반적 동작 retargeting, 실제 로봇 전이는 미검증.",
                "문제·근거·수정 완료 조건: 같은 폴더의 CURRENT_ISSUES.md"],seconds=4)
        finally:movies.append(movie.close())

    successful=[case for case in cases if case["case_id"].startswith("fast_bc_before_") and case["terminal"]["native_success"]]
    movie=Movie(videos/"03_fast_success_gallery.mp4")
    try:
        movie.card("fast 학습 정책 성공 사례 전체",[
            f"초기 평가 10회 중 성공했던 {len(successful)}회 모두 재실행.",
            "동일한 저장 모델 사용. 시드별 초기 배치만 달라짐.",
            "각 장면은 목표 칸 표시와 같은 정책의 1초 추가 확인을 포함.",
            "실패한 seed10000은 별도의 실패 비교 영상에 포함."],seconds=5)
        for start in range(0,len(successful),2):
            group=successful[start:start+2]
            movie.scene(f"성공 사례 {start+1}–{min(start+2,len(successful))} / {len(successful)}",[(f"seed{case['seed']} | stable1s={case['stable_placement_1s']}",output/case["video_path"]) for case in group],
                        notes=["데이터 생성 교사가 아닌, 저장된 모방학습 신경망의 출력으로 제어.","현재 입력은 시뮬레이터 상태와 규칙 기반 단계 정보이며 영상 입력이 아님."])
    finally:movies.append(movie.close())
    movie=Movie(videos/"04_failure_teacher_vs_policy.mp4")
    try:
        movie.card("현재 핵심 실패 | 같은 seed10000",[
            "왼쪽 교사와 오른쪽 학습 정책의 초기 특징값이 동일한지 검사.",
            "교사는 배치에 성공하지만 학습 정책은 접근 단계를 완료하지 못함.",
            "unreachable_waypoint 표기만으로 물리적 도달 불가능을 주장할 수 없음.",
            "이 진단용 교사 실행은 평가 전용이며 학습 데이터에 넣지 않음."],seconds=5)
        movie.scene("같은 조건: 교사 성공 / 학습 정책 실패",[("교사 seed10000",clip("fast_teacher_10000")),("학습 정책 seed10000",clip("fast_bc_before_10000"))],
                    notes=["관측된 실패 위치: pregrasp. 근본 원인은 아직 확정하지 않음.","교사와 정책의 행동·정규화·관측 분포를 비교해야 함."])
        movie.scene("추가 데이터 후에도 남은 같은 실패",[("피드백 전",clip("fast_bc_before_10000")),("피드백 후",clip("fast_bc_after_10000"))],
                    notes=["무작위 새 교사 시작점을 추가하는 기존 개입은 이 실패를 해결하지 못함.","현재 문제 명세의 P01·P04·P05 참고."])
    finally:movies.append(movie.close())

    same_seed=by_id["fast_bc_before_10000"]
    teacher=by_id["fast_teacher_10000"]
    with np.load(raw/"rollouts/fast_bc_before_10000.npz") as a,np.load(raw/"rollouts/fast_teacher_10000.npz") as b:
        initial_match=bool(np.allclose(a["features"][0],b["features"][0],atol=1e-7))
    if not initial_match:raise ValueError("Teacher/failure comparison uses different initial features")
    media=verify_media(output)
    manifest={"schema_version":"1.0","source_runs":[FAST,BOOT],"human_video_used":False,"new_training_performed":False,
              "observation_mode":"oracle_sim_state","placement_stability_check":"same original controller continued for 1s; specific Can in target and no grasp contact",
              "fast_initial_success_replays":len(successful),"fast_initial_stable_success_replays":sum(c["stable_placement_1s"] for c in successful),
              "teacher_vs_failed_policy_initial_features_match":initial_match,"movies":movies,"verified_media":media,
              "limitations":["Full human-video branch untested","Source transfer uses grasp offset and lift geometry plus hand-coded phases",
                             "Current success counts do not establish causal source-transfer advantage","10 fast evaluation seeds only"]}
    (output/"media_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print(f"[review] assembled {len(movies)} overview movies and verified {len(media)} MP4 files",flush=True)
