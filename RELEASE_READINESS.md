# Release Readiness Notes

This file records what is ready for public review and what is intentionally excluded.

## Ready for Review

- Source code for dataset preparation, training, pseudo-label filtering, calibration, and reporting.
- Configuration template under `configs/config.yaml`.
- Selected CSV/JSON formal reports under `reports/`.
- Report-to-manuscript mapping under `reports/REPORT_MANIFEST.md`.
- JAE manuscript source and compiled PDF under `manuscript_jae/`.
- Submission support files under `submission_jae/`.
- Reproducibility guide under `REPRODUCIBILITY.md`.
- Citation metadata under `CITATION.cff`.

## Intentionally Excluded

- Raw third-party datasets.
- Generated prepared dataset copies.
- Full YOLO run folders.
- Trained weights and model exports.
- Local machine paths and authentication files.

These exclusions are deliberate. The raw datasets and trained weights may be subject to third-party license or access conditions, and the full experiment folders are too large for a manuscript-support repository.

## Submission Readiness

The repository currently supports the following manuscript claims:

- Source-only training on video-frame data has poor locked external-domain transfer.
- Class-balanced target-domain adaptation is the primary fair baseline.
- Strict SSOD is a high-precision operating point rather than a large mAP-only improvement.
- External evaluation uses a four-class masked protocol because the external dataset lacks `empty` and `overripe`.
- Third-domain outdoor evaluation is exploratory and image-level.

## Recommended Action Before Final Publication

If the manuscript is accepted, archive a frozen release on GitHub and, if possible, generate a DOI through Zenodo or an institutional repository. Do not upload raw datasets or trained weights unless the original data providers' licenses permit redistribution.
