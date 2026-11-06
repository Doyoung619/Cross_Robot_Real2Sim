# English technical report

[Open the 19-page PDF](Cross_Robot_Real2Sim_Technical_Report.pdf).

The report explains the models, data, algorithms, interfaces, measured outcomes, problems, and scientific limits of all five pipeline stages. It includes the actual source-selection algorithm, manual task specification, source grasp/lift extraction, eight-phase target controller, 26D features, 4D-to-7D action adapter, BC loss/checkpoint protocol, held-out evaluation, diagnostic rules, and measured feedback results.

The three English case studies share **one baseline UR5e policy** trained on ten executed teachers derived from five Panda source IDs. Their original-source videos reconstruct recorded HDF5 states; they do not execute old recorded actions. The report separates these videos from the earlier scripted Panda smoke branch and from actual UR5e teacher/policy rollouts. CLIP/Qwen inference, SAM2, real human-video success, and real-robot transfer are not claimed.

## Rebuild

From the repository root:

```bash
.venv/bin/python -m pip install -r requirements-report.txt
.venv/bin/python scripts/build_english_report.py
```

The build reads the editable [structured English source](report_source.json), preserved source-replay metadata, and three existing English videos for illustrative stills. It does not retrain a policy or rerun robot simulation. [The build manifest](build_manifest.json) records input and PDF hashes. Exact installed packages for this report build are recorded in `requirements-working-report.lock` at the repository root.

## Verification files

- [QA report](qa/qa_report.json): 19 pages, all-English extracted text, embedded DejaVu fonts, no text outside margins, and no page overflow.
- [Contact sheet 1](qa/contact_sheet_01.png), [2](qa/contact_sheet_02.png), [3](qa/contact_sheet_03.png), [4](qa/contact_sheet_04.png): all pages rendered for visual review.
- [Extracted text](qa/extracted_text.txt) and [flowable layout audit](qa/layout_audit.json).
- [Case-study page](qa/qa_page_16.png) and [source-validation page](qa/qa_page_17.png): larger rendered samples.
- [Figure provenance](figures/frame_provenance.json): original video hashes, frame indices, and source filenames.

The fonts are embedded TrueType subsets with Unicode maps; the PDF contains no default unembedded Helvetica resource. The builder uses ReportLab for layout, pypdf for font inspection, PyMuPDF for text bounds/rendering, and `pdftotext` for a second extraction check. PDF quality checks do not add robotics performance evidence.
