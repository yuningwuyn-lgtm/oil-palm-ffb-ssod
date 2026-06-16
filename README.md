# Quality-Controlled SSOD for Cross-Domain Oil Palm FFB Maturity Detection

This repository contains a Python/YOLOv8 research pipeline for oil palm fresh fruit bunch (FFB) maturity detection under cross-domain evaluation.

The project is positioned as:

> A quality-controlled semi-supervised YOLOv8 framework for cross-domain oil palm FFB maturity detection.

## Main Idea

Source-domain oil palm FFB detectors can perform well on same-source video-frame data, but may fail when evaluated on another dataset or plantation scene. This project therefore evaluates oil palm maturity detection using stricter protocols:

- scene/video-disjoint source-domain evaluation;
- locked external-domain evaluation;
- four-class masked evaluation for external datasets without `empty` and `overripe`;
- duplicate checking;
- class-space harmonization;
- quality-controlled pseudo-label self-training.

## Repository Layout

```text
src/
  full_ssod_ffb_pipeline.py          Main end-to-end SSOD pipeline
  train_model2_adaptation.py         Target-domain adaptation runner
  run_protocol_v2_formal_experiments.py
  run_precision_first_ssod_formal.py
  run_third_domain_zero_shot_formal.py
  evaluate_locked_external_protocol.py
  paper_reporting.py

configs/
  config.yaml

reports/
  Selected CSV/JSON outputs from formal experiments

manuscript_jae/
  LaTeX manuscript draft and figures for Journal of Agricultural Engineering

DATASETS.md
  Dataset setup and class-space notes
```

## Key Experimental Stages

| Stage | Purpose |
|---|---|
| Source-only YOLOv8n | Measures source-domain training and external-domain collapse |
| Model2 Balanced | Class-balanced target-domain adaptation baseline |
| Strict SSOD | Precision-first pseudo-label self-training with quality control |

## Main Formal Results

Three-seed formal comparison:

| Model | Scene mAP50 | External mAP50 | External mAP50-95 | External Precision | External Recall | External F1 |
|---|---:|---:|---:|---:|---:|---:|
| Source-only YOLOv8n | 0.5917 | 0.0372 | 0.0130 | 0.1640 | 0.0933 | 0.0735 |
| Model2 Balanced | 0.6459 | 0.6973 | 0.4758 | 0.6126 | 0.7527 | 0.6442 |
| Strict SSOD calibrated | 0.6373 | 0.7240 | 0.4863 | 0.8753 | 0.5364 | 0.6321 |

Interpretation:

- The source-only result demonstrates strong cross-domain shift.
- Model2 Balanced demonstrates target-domain adaptation, not zero-shot generalization.
- Strict SSOD mainly improves the high-precision operating point.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Example Usage

Run the full SSOD pipeline with local datasets:

```powershell
python src/full_ssod_ffb_pipeline.py `
  --max-epochs 10 `
  --self-training-iterations 1 `
  --workers 0
```

Run formal protocol scripts after configuring local dataset paths:

```powershell
python src/run_protocol_v2_formal_experiments.py
python src/run_precision_first_ssod_formal.py
python src/run_third_domain_zero_shot_formal.py
```

## Manuscript

A LaTeX manuscript draft for Journal of Agricultural Engineering is included under:

```text
manuscript_jae/main.tex
manuscript_jae/main.pdf
```

Before submission, update:

- dataset/code availability links;
- funding statement;
- acknowledgements;
- final reference formatting;
- author affiliation if needed.

## Data and Weights

Datasets, trained weights, generated prepared datasets, and large experiment folders are intentionally excluded from this repository. See `DATASETS.md`.

## License

TODO: choose a license before making the repository public. MIT is suitable for code, but dataset licenses must be checked separately.

