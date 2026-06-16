# Reproducibility Guide

This document explains how to reproduce the manuscript workflow from local datasets. The repository intentionally excludes raw images, trained weights, prepared dataset copies, and full YOLO run folders because those artifacts are large and may be governed by third-party dataset licenses.

## 1. Environment

Recommended environment:

- Windows 10/11 or Linux with Python 3.10+
- NVIDIA GPU for training
- Ultralytics YOLOv8
- `workers=0` on Windows when using Ultralytics training

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Quick syntax check:

```powershell
Get-ChildItem -Path src -Filter *.py -File | ForEach-Object {
  python -m py_compile $_.FullName
}
```

## 2. Required Local Data

Prepare local copies of the datasets described in `DATASETS.md`:

| Dataset role | Required for | Notes |
|---|---|---|
| Source-domain ScienceDB/OILPALM-style FFB dataset | Source-only baseline and scene-disjoint evaluation | Use video/scene-disjoint splitting; do not use random split as the main conclusion. |
| External preprocessed FFB dataset | Model2 Balanced adaptation and locked external testing | Use the four-class masked protocol: `abnormal`, `ripe`, `under_ripe`, `unripe`; ignore `empty` and `overripe`. |
| Weakly labeled or unlabeled new images | SSOD pseudo-label generation | Folder names may be used only as weak image-level priors. |
| Outdoor oil palm ripeness dataset | Third-domain zero-shot image-level evaluation | Not used for training. Report as exploratory image-level evaluation. |

Keep the locked external test split out of all training, hard-negative mining, pseudo-label generation, and threshold tuning.

## 3. Configuration

Use `configs/config.yaml` as the central template for dataset roles, class names, and protocol assumptions.

Several formal runner scripts contain sanitized placeholders such as:

- `LOCAL_PROJECT_ROOT`
- `LOCAL_DATA_ROOT`
- `PATH/TO/...`

Before rerunning the full experiments, replace those placeholders with local absolute paths to your prepared datasets and output folders. This is deliberate: machine-specific private paths were removed before publishing the repository.

## 4. Main Formal Experiment Commands

Run the formal source/adaptation comparison:

```powershell
python src/run_protocol_v2_formal_experiments.py `
  --seeds 42,2026,3407 `
  --epochs 10 `
  --imgsz 640 `
  --batch 4
```

Expected report:

```text
reports/protocol_v2_formal/formal_seed_results.csv
```

In the public repository, the selected result file is archived as:

```text
reports/paper_framework_reports_protocol_v2_formal_formal_seed_results.csv
```

Run the precision-first strict SSOD formal comparison:

```powershell
python src/run_precision_first_ssod_formal.py `
  --seeds 42,2026,3407 `
  --epochs 10 `
  --imgsz 640 `
  --batch 4
```

Expected archived result:

```text
reports/paper_framework_reports_precision_first_ssod_formal_strict_ssod_formal_final_seed_results.csv
```

Run the third-domain zero-shot image-level evaluation:

```powershell
python src/run_third_domain_zero_shot_formal.py
```

Expected archived results:

```text
reports/paper_framework_reports_third_domain_outdoor_zenodo_formal_third_domain_zero_shot_seed_results.csv
reports/paper_framework_reports_third_domain_outdoor_zenodo_formal_third_domain_zero_shot_summary.csv
```

## 5. Manuscript Result Mapping

Use `reports/REPORT_MANIFEST.md` to map archived CSV/JSON outputs to the manuscript tables and claims.

Core interpretation rules:

- Source-only YOLOv8n is domain-shift evidence, not the main improvement baseline.
- Model2 Balanced is the primary supervised target-domain adaptation baseline.
- Strict SSOD is a conservative high-precision operating point.
- Third-domain evaluation is image-level and exploratory; it is not directly comparable to detection mAP.
- External preprocessed FFB evaluation masks classes absent from that dataset.

## 6. Manuscript Build

The JAE-oriented manuscript is in:

```text
manuscript_jae/main.tex
manuscript_jae/main.pdf
```

Compile with `pdflatex`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The prepared manuscript follows the Journal of Agricultural Engineering Original Article constraints:

- one PDF manuscript file;
- 12-point, double-spaced text;
- unstructured abstract below 400 words;
- references below 40;
- total tables and figures below 15;
- tables and figures placed at the end.

## 7. Submission Materials

Submission-support files are in:

```text
submission_jae/
```

Included files:

- `cover_letter.md`
- `editor_comments.md`
- `submission_checklist.md`
- `suggested_reviewers_template.csv`
- `jae_format_audit.md`

Before final submission, manually confirm that the online submission form still accepts the prepared PDF and that author affiliation details are correct.
