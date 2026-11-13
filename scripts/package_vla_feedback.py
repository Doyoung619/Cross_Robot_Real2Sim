#!/usr/bin/env python3
"""Package measured failure-feedback rounds without inventing missing outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def round_order(path):
    name = path.parent.name
    if name == "baseline":
        return (-1, name)
    numbers = re.findall(r"\d+", name)
    return (int(numbers[-1]) if numbers else 999, name)


def discover(raw):
    paths = sorted((raw / "rounds").glob("*/evaluation.json"), key=round_order)
    paths += sorted(raw.glob("final*/evaluation.json"))
    paths += sorted(raw.glob("control*/evaluation.json"))
    result = []
    for path in paths:
        payload = read_json(path, [])
        rows = payload if isinstance(payload, list) else payload.get("rows", payload.get("episodes", []))
        if not rows:
            continue
        parent = path.parent
        role = "development" if parent.parent.name == "rounds" else ("control" if parent.name.startswith("control") else "final")
        conditions = sorted({r.get("condition", "policy") for r in rows}) if role != "development" else [None]
        for condition in conditions:
            subset = [r for r in rows if r.get("condition", "policy") == condition] if condition is not None else rows
            result.append(dict(name=parent.name + (":" + str(condition) if condition is not None else ""),
                           path=parent, final=role == "final", role=role, rows=subset,
                           diagnosis=read_json(parent / "diagnosis.json", {}),
                           selection=read_json(parent / "selection.json", {}),
                           processing=read_json(parent / "processing_audit.json", {}),
                           training=read_json(parent / "training.json", {}),
                           intervention=read_json(parent / "intervention.json", {})))
    return result


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def mean(rows, key):
    values = [float(r[key]) for r in rows if number(r.get(key))]
    return sum(values) / len(values) if values else None


def metric(value, scale=1, suffix="", decimals=2):
    return f"{value*scale:.{decimals}f}{suffix}" if value is not None else "not recorded"


def source_ids(entry):
    selection = entry["selection"]
    values = next((selection[k] for k in ("selected", "selected_source_ids", "selected_source_demo_ids", "source_demo_ids") if k in selection), [])
    if isinstance(values, list):
        return [v.get("demo_id", str(v)) if isinstance(v, dict) else str(v) for v in values]
    return [str(values)] if values else []


def describe_change(entry):
    intervention = entry["intervention"]
    if isinstance(intervention, str):
        return intervention
    for key in ("description", "summary", "action", "method"):
        if intervention.get(key):
            return str(intervention[key])
    for block in (entry["selection"], entry["processing"]):
        for key in ("feedback_method", "transformation", "retargeting_method", "method"):
            if block.get(key):
                value = block[key]
                return json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list)) else str(value)
    return "Baseline adapter; no feedback update" if entry["name"] == "baseline" else "See per-round selection, processing and diagnosis metadata"


def copy_artifacts(raw, dest, entries):
    for name in ("summary.json", "request.json", "status.json", "progress.json", "protocol.json", "data_verification.json", "INPUT_CONTRACT_CORRECTION.md"):
        if (raw / name).is_file():
            shutil.copyfile(raw / name, dest / name)
    for entry in entries:
        relative = entry["path"].relative_to(raw)
        target = dest / relative
        target.mkdir(parents=True, exist_ok=True)
        for source in entry["path"].rglob("*"):
            if source.is_file() and source.suffix in (".json", ".csv", ".safetensors"):
                output = target / source.relative_to(entry["path"])
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, output)
        for row in entry["rows"]:
            if not row.get("video"):
                continue
            source = entry["path"] / row["video"]
            if not source.is_file():
                raise FileNotFoundError(source)
            output = target / row["video"]
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)


def make_montage(raw, dest, entries, scene):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    selected = []
    for entry in entries:
        if entry["role"] != "development":
            continue
        match = next((r for r in entry["rows"] if r.get("source_initial_scene_id") == scene and r.get("video")), None)
        if match:
            selected.append((entry, match))
    if not selected:
        return None
    readers = [imageio.get_reader(e["path"] / r["video"]) for e, r in selected]
    lengths = [r.count_frames() for r in readers]
    fps = min(float(r.get_meta_data().get("fps", 10)) for r in readers)
    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title = ImageFont.truetype(fontpath, 32)
    body = ImageFont.truetype(fontpath, 21)
    small = ImageFont.truetype(fontpath, 17)
    columns = min(len(selected), 4)
    rows_count = (len(selected) + columns - 1) // columns
    # H.264 yuv420 requires even dimensions; keep a conventional 720px height
    # for the expected single row of four round panels.
    width, height = columns * 640, 144 + rows_count * 480 + 96
    output = dest / "videos" / f"feedback_rounds_{scene}.mp4"
    output.parent.mkdir(exist_ok=True)
    try:
        with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=7, macro_block_size=1) as writer:
            for index in range(max(lengths)):
                image = Image.new("RGB", (width, height), (12, 20, 32))
                draw = ImageDraw.Draw(image)
                draw.text((25, 20), "Failure feedback | source reselection + offline retargeting + VLA updates", font=title, fill="white")
                draw.text((25, 70), f"Development scene {scene}: these rollouts guide interventions; they are not independent test results.", font=body, fill=(190, 216, 240))
                for j, ((entry, row), reader) in enumerate(zip(selected, readers)):
                    x, y = (j % columns) * 640, 135 + (j // columns) * 465
                    draw.text((x + 15, y), entry["name"], font=body, fill="white")
                    frame = Image.fromarray(reader.get_data(min(index, lengths[j] - 1))).resize((640, 360))
                    image.paste(frame, (x, y + 35))
                    success = bool(row.get("success", False))
                    outcome = "SUCCESS" if success else "FAILURE"
                    draw.text((x + 15, y + 405), f"{outcome} | goal error {metric(row.get('goal_error_m'),100,' cm')}", font=body,
                              fill=(110, 225, 160) if success else (255, 185, 130))
                    draw.text((x + 15, y + 438), f"Plate-contact fraction: {metric(row.get('contact_fraction'),100,'%')}", font=small, fill="white")
                draw.text((25, height - 72), "Policy-only live physics rollouts. Static source-derived training renders are not shown as successful executions.", font=body, fill=(190, 216, 240))
                draw.text((25, height - 35), "Lower goal distance is not task success. See separate reserved final-scene evaluation in the report.", font=small, fill=(255, 210, 150))
                writer.append_data(np.asarray(image))
        figures = dest / "figures"
        figures.mkdir(exist_ok=True)
        with imageio.get_reader(output) as reader:
            imageio.imwrite(figures / f"feedback_rounds_{scene}_last.png", reader.get_data(reader.count_frames() - 1))
    finally:
        for reader in readers:
            reader.close()
    return output.relative_to(dest).as_posix()


def measured_table(entries):
    lines = ["| Stage | Role | Success / episodes | Mean final XY error | Mean plate-contact fraction |",
             "|---|---|---:|---:|---:|"]
    for e in entries:
        rows = e["rows"]
        role = {"final": "Reserved final evaluation", "control": "Compute-matched control", "development": "Development feedback"}[e["role"]]
        lines.append(f"| {e['name']} | {role} | "
                     f"{sum(bool(r.get('success')) for r in rows)}/{len(rows)} | {metric(mean(rows,'goal_error_m'),100,' cm')} | "
                     f"{metric(mean(rows,'contact_fraction'),100,'%')} |")
    return lines


def report_markdown(raw, dest, entries, montage_paths):
    text = ["# B VLA failure-feedback experiment", "",
            "Task: push the plate to the front of the stove. Starting point: the preserved 100-update B LoRA adapter, not the ten-minute continuation.", "",
            "B failure rollouts now provide diagnostic feedback for choosing existing A source data and changing offline retargeting/action labels. This is a different protocol from the earlier source-only, evaluation-only experiment.", "",
            "## Measured round-by-round outcomes", ""]
    text += measured_table(entries)
    text += ["", "Development scenes are intentionally reused to guide the feedback loop. Their improvements are development measurements, not independent generalization evidence. Reserved final scenes are listed separately and should not guide interventions.", "",
             "## What each feedback round changed", ""]
    for e in entries:
        if e["name"] == "baseline" or e["role"] != "development":
            continue
        relative = e["path"].relative_to(raw).as_posix()
        text += [f"### {e['name']}", "", describe_change(e), ""]
        ids = source_ids(e)
        if ids:
            text += [f"Selected source IDs: {', '.join(ids)}.", ""]
        if e["training"]:
            m = e["training"]
            steps = m.get("additional_steps", m.get("updates", m.get("steps", "not recorded")))
            text += [f"Recorded update count: {steps}. Exact training provenance is in [{relative}/training.json]({relative}/training.json).", ""]
        if e["diagnosis"]:
            text += [f"[Measured rollout diagnosis]({relative}/diagnosis.json). These are evidence-based failure hypotheses, not proven causal diagnoses.", ""]
    training = [e["training"] for e in entries if e["training"]]
    counts = [m.get("executed_target_training_episodes", m.get("executed_B_training_demonstrations", m.get("target_supervised_episodes"))) for m in training]
    if counts and all(number(v) and v == 0 for v in counts):
        text += ["No executed B trajectory is used as action supervision in the recorded updates. B rollout diagnostics influence source selection and offline processing; the source-derived frames and their labels remain the supervised data.", ""]
    text += ["## English comparison movies", ""]
    for path in montage_paths:
        text.append(f"- [{Path(path).stem}]({path})")
    text += ["", "## Individual live rollouts", ""]
    for e in entries:
        for row in e["rows"]:
            if row.get("video"):
                video = (e["path"].relative_to(raw) / row["video"]).as_posix()
                text.append(f"- [{e['name']} / {row.get('source_initial_scene_id','scene')} / success={bool(row.get('success'))}]({video})")
    control_text = ("A compute-matched no-feedback control is reported separately, which helps compare feedback-guided processing with additional updates on unchanged data. Selection and retargeting still change together, so their individual contributions are not isolated."
                    if any(e["role"] == "control" for e in entries) else
                    "No matched no-feedback control is available, so this run cannot isolate feedback from additional training.")
    text += ["", "## Interpretation limits", "",
             "Each intervention changes data selection, offline labels and adapter weights together. " + control_text + " A development improvement does not establish broad transfer, and a failed intervention does not disprove every proposed hypothesis.", "",
             "Offline IK and static rendering do not validate target pushing friction or contact dynamics. The physically executed policy rollouts and native task success are the actual execution evidence. Trace, diagnosis, selection and processing files are preserved per round.", "",
             "Retargeting uses a bounded static contact-offset search at three source poses and a smooth trajectory correction, not dynamics-aware trajectory optimization. New action labels use future TCP displacement over a response-conditioned 2–5-frame lookahead; this is heuristic controller inversion and also changes action magnitude. It is not a calibrated inverse of the OSC dynamics.", "",
             "[English PDF](FEEDBACK_REPORT.pdf)."]
    if (raw / "INPUT_CONTRACT_CORRECTION.md").is_file():
        text += ["", "## Preserved correction record", "",
                 "A first execution attempt was corrected before the reported round comparison. Its documented correction is retained in [INPUT_CONTRACT_CORRECTION.md](INPUT_CONTRACT_CORRECTION.md); the archived attempt is not silently mixed into the current scores."]
    result = "\n".join(text) + "\n"
    (dest / "FINAL_REPORT.md").write_text(result)
    (dest / "README.md").write_text(result)


def pdf_report(dest, entries):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors

    pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVuBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    output = dest / "FEEDBACK_REPORT.pdf"
    pdf = canvas.Canvas(str(output), pagesize=A4, initialFontName="DejaVu", initialFontSize=10)
    width, height = A4
    style = ParagraphStyle("body", fontName="DejaVu", fontSize=10, leading=15)
    y = height - 50

    def page(title):
        nonlocal y
        if y != height - 50:
            pdf.showPage()
        y = height - 50
        pdf.setFont("DejaVuBold", 16)
        pdf.drawString(42, y, title)
        y -= 35

    def paragraph(text):
        nonlocal y
        obj = Paragraph(html.escape(text), style)
        _, h = obj.wrap(width - 84, height)
        if y - h < 50:
            page("Failure-feedback experiment (continued)")
        obj.drawOn(pdf, 42, y - h)
        y -= h + 12

    page("B VLA failure-feedback experiment")
    paragraph("Task: push the plate to the front of the stove. This experiment starts from the original 100-update B adapter. B failures guide existing A source-data selection and offline retargeting, followed by short LoRA updates and new live policy rollouts.")
    data = [["Stage / role", "Success", "Mean error", "Contact fraction"]]
    for e in entries:
        rows = e["rows"]
        data.append([Paragraph(html.escape(e["name"] + " / " + e["role"]), style),
                     f"{sum(bool(r.get('success')) for r in rows)}/{len(rows)}",
                     metric(mean(rows, "goal_error_m"), 100, " cm"),
                     metric(mean(rows, "contact_fraction"), 100, "%")])
    table = Table(data, colWidths=[180, 60, 105, 165])
    table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f4")),
                               ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#8da3b9")),
                               ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    _, h = table.wrap(width - 84, height)
    table.drawOn(pdf, 42, y - h)
    y -= h + 20
    paragraph("Development scores are used by the feedback loop and are not independent test results. Reserved final evaluation, when present, is reported separately. Native task success remains distinct from merely reducing goal distance.")
    for e in entries:
        if e["name"] == "baseline" or e["role"] != "development":
            continue
        paragraph(e["name"] + ": " + describe_change(e))
        ids = source_ids(e)
        if ids:
            paragraph("Source selection: " + ", ".join(ids))
    page("Implemented selection and retargeting")
    paragraph("1. Data selection. The original supervised anchors come from successful A demonstrations demo_5, demo_38 and demo_26. Each feedback round searches unused successful A demonstrations with IDs below 40; IDs 40-49 are excluded from all added training sources. It adds two demonstrations after B inverse-kinematics and static arm-collision checks.")
    paragraph("The source retrieval score is 2 x push-direction alignment + 3 x source plate displacement + 2 x rear-contact offset + 0.002 x estimated contact-window length - 0.001 x trajectory length. Direction alignment is the dot product of the source plate-motion direction and B's measured remaining-goal direction. Rear contact is the negative projection of the source EEF-to-plate offset onto that direction. Distances are in metres; lengths are source frames. The contact window is inferred from plate motion, not measured source contact forces.")
    added = []
    for e in entries:
        if e["role"] == "development" and e["name"] != "baseline":
            ids = e["processing"].get("selected_new_source_ids", e["selection"].get("feedback_selected", []))
            added.append(e["name"] + ": " + ", ".join(map(str, ids)))
    if added:
        paragraph("New source demonstrations by round: " + "; ".join(added) + ".")
    paragraph("2. Task context. The fixed language instruction is to push the plate to the front of the stove. The simulator supplies the plate and goal geometry for diagnosis. This short experiment does not train a separate task-understanding model or infer the task from a human video.")
    paragraph("3. Offline retargeting. A bounded candidate-offset grid is evaluated at three source contact-window poses. The static objective uses finger-to-plate signed distance, penetration, IK error, unintended arm collision and correction magnitude. XY correction is bounded by 2 cm and height by 1 cm. A smooth contact-window correction is applied before solving the source TCP trajectory for B and rendering B's training views. No target physics rollout is executed to manufacture action labels.")
    paragraph("All 50 future actions per new training anchor are recomputed from achieved B TCP poses. Translation uses future displacement divided by the controller's 0.05 m scale; rotation uses the relative rotation divided by 0.5 rad; gripper labels come from A. The lookahead is clipped to 2-5 frames using the reciprocal measured plate-response gain. This is heuristic controller inversion, not a calibrated inverse OSC or dynamics-aware trajectory optimization. Padding masks exclude invalid current or lookahead IK endpoints. Static geometry does not establish valid pushing friction or force response.")
    page("Training, feedback and final evaluation")
    paragraph("4. Policy learning. The pretrained lerobot/smolvla_libero SmolVLA backbone (about 450 million parameters) remains frozen. Rank-8 LoRA adapters contain 334,592 trainable parameters. Starting from the preserved 100-update adapter, each round performs 100 additional gradient updates with batch size 4, learning rate 0.0001 and masked action-regression loss. Adapter weights continue between rounds; Adam state is restored for rounds 2 and 3. These counts are optimizer updates, not full dataset epochs.")
    sizes = [e["training"].get("training_frames") for e in entries if e["role"] == "development" and e["name"] != "baseline"]
    sizes = [str(x) for x in sizes if number(x)]
    if sizes:
        paragraph("Supervised anchor counts grow from the original 189 to " + ", then ".join(sizes) + ". Each feedback minibatch draws half its anchors from the previous source data and half from the newly processed corrective source data. The original 189 images, states and actions remain bitwise unchanged; data_verification.json records this audit and zero invalid unmasked lookahead endpoints.")
    paragraph("5. Failure feedback. Live B policy rollouts measure native success, final plate-goal XY error, gripper-plate contact fraction, closest-surface vectors, push-direction alignment and plate response. A heuristic diagnosis proposes contact geometry, direction or dynamics/action-scale candidates. Those measured diagnostics affect the next A retrieval query and offline correction/lookahead. B rollout actions are never supervised targets, and zero executed B demonstrations are used for training. These hypotheses are not causal proofs.")
    summary = read_json(dest / "summary.json", {})
    selected_round = summary.get("best_development_round", "recorded in summary.json")
    paragraph("Development IDs demo_40 and demo_41 are reused for feedback. Round " + str(selected_round) + " was selected using development success count and then mean goal error, before evaluating reserved final IDs demo_47 and demo_48. The final table reports the original and selected adapters separately on the same reserved scenes. The no-feedback control receives 300 additional updates on unchanged original data with the same training hyperparameters; it is evaluated on the development scenes.")
    final_entries = [e for e in entries if e["role"] == "final"]
    if final_entries:
        paragraph("Reserved final outcomes: " + "; ".join(e["name"] + " = " + str(sum(bool(r.get("success")) for r in e["rows"])) + "/" + str(len(e["rows"])) + " successes, mean final XY error " + metric(mean(e["rows"], "goal_error_m"), 100, " cm") for e in final_entries) + ". Two reserved scenes are a small pilot check, not a precise generalization estimate.")
    page("Evidence, protocol and limits")
    paragraph("The per-round files preserve actual live rollout traces, contact measurements, diagnosis hypotheses, selected source IDs, offline transformations and training metadata. The English montage movies compare the baseline and all recorded development rounds, including failures.")
    paragraph("Source reconstructions and static B rendering are training-data processing. They do not prove valid B contact dynamics. Only the live policy rollouts establish what the learned B policy physically does in simulation.")
    paragraph("Rounds change multiple factors together. A compute-matched control, when present in the table, compares the combined feedback changes against additional unchanged-data training; it does not isolate selection from retargeting. A small repeated development set does not establish generalization; final reserved scenes are the available check against development overfitting.")
    paragraph("A measured low contact fraction, poor approach geometry or direction mismatch can justify a candidate intervention. It is not proof of friction or dynamics being the cause. The report deliberately distinguishes observed measurements, hypotheses and implemented changes.")
    if (dest / "INPUT_CONTRACT_CORRECTION.md").is_file():
        paragraph("A first execution attempt required a documented correction. See INPUT_CONTRACT_CORRECTION.md. The archived attempt is excluded from this table, rather than silently combined with corrected-run measurements.")
    pdf.save()
    return output


def verify(dest):
    import fitz
    pdf = fitz.open(dest / "FEEDBACK_REPORT.pdf")
    qa = dest / "qa"
    qa.mkdir(exist_ok=True)
    pages = []
    for i, page in enumerate(pdf):
        text = page.get_text()
        assert "\ufffd" not in text
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    assert min(x0, y0) >= 0 and x1 <= page.rect.width + .5 and y1 <= page.rect.height + .5
        page.get_pixmap(matrix=fitz.Matrix(1, 1)).save(qa / f"page_{i+1:02}.png")
        pages.append(dict(page=i+1, text_characters=len(text), layout_passed=True))
    videos = []
    for movie in dest.rglob("*.mp4"):
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(movie)], text=True))
        assert float(probe["format"]["duration"]) > 0
        videos.append(str(movie.relative_to(dest)))
    (qa / "verification.json").write_text(json.dumps(dict(status="passed", pdf_pages=pages, videos=videos), indent=2))


def package(raw, dest=None):
    raw = Path(raw).resolve()
    dest = Path(dest).resolve() if dest else ROOT / "results" / raw.name
    dest.mkdir(parents=True, exist_ok=True)
    entries = discover(raw)
    if not entries:
        raise RuntimeError("No measured evaluation rows are available to package")
    copy_artifacts(raw, dest, entries)
    scenes = sorted({r.get("source_initial_scene_id") for e in entries if e["role"] == "development" for r in e["rows"] if r.get("source_initial_scene_id")})
    montage_paths = [path for scene in scenes if (path := make_montage(raw, dest, entries, scene))]
    report_markdown(raw, dest, entries, montage_paths)
    pdf_report(dest, entries)
    verify(dest)
    files = [dict(path=str(p.relative_to(dest)), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(dest.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"]
    (dest / "artifact_manifest.json").write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(), files=files), indent=2))
    print(dest)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    package(args.run_dir, args.output)
