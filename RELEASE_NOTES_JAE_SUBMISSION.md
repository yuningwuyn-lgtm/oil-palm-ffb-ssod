# Release Notes: JAE Submission Snapshot

Version: `v1.0.1-jae-author-guidelines`

Date: 2026-06-16

Repository: `https://github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod`

## Purpose

This release freezes the manuscript-support repository for Journal of Agricultural Engineering submission review. It provides a stable snapshot of the manuscript, source code, selected formal reports, reproducibility documentation, and submission-support materials.

## Included

- JAE-oriented manuscript source and compiled PDF.
- Cover letter, editor comments, final submission package map, submission form text, reviewer list, checklist, and artifact checksums.
- Source code for dataset preparation, training orchestration, pseudo-label filtering, external evaluation, calibration, and reporting.
- Selected formal CSV/JSON reports used to support manuscript tables and claims.
- Report manifest mapping archived results to manuscript tables and interpretation rules.
- Reproducibility guide and public release-readiness notes.
- Citation metadata through `CITATION.cff`.
- GitHub Actions workflow for lightweight submission-package validation.
- JAE author-guideline updates: author-year citation style, 3-6 alphabetized keywords, full postal address, and generative AI declaration.

## Excluded

- Raw third-party datasets.
- Trained weights.
- Full YOLO run folders.
- Generated prepared dataset copies.
- Local machine paths and authentication files.

These exclusions are deliberate because raw data and trained weights may be governed by the original data providers' license/access conditions and because full experiment outputs are too large for a manuscript-support repository.

## Validated State

The validator checks:

- Manuscript PDF exists and matches the recorded checksum.
- Abstract length is below 400 words.
- References are below 40.
- Tables plus figures are below 15.
- Citation keys and bibliography entries are consistent.
- JAE keyword count/order, author-year citation commands, postal address, and generative AI declaration are checked.
- Suggested reviewers are present with valid email fields.
- Submission checklist has no open items.
- PDF has no blank-like pages when PyMuPDF is available.
- Python source files compile.
- Known local/private path strings are absent.

Latest validated manuscript PDF:

- File: `manuscript_jae/main.pdf`
- Pages: 20
- Size: 467092 bytes
- SHA256: `7F0580FFCC2460D84B7DF2EB9D811CC9D2579BFD8E777EFCDD8BB167E8B812D3`

## Interpretation Reminder

- Source-only YOLOv8n is interpreted as domain-shift evidence.
- Model2 Balanced is the main fair target-domain adaptation baseline.
- Strict SSOD is interpreted as a high-precision operating point.
- External preprocessed FFB evaluation uses a four-class masked protocol.
- Third-domain outdoor evaluation is exploratory and image-level.
