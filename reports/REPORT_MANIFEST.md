# Report Manifest

This folder contains selected lightweight CSV/JSON outputs used to support the manuscript tables and figures. Raw datasets, trained weights, prepared dataset copies, and full YOLO run folders are excluded from Git.

## Formal Three-Seed Detection Results

| File | Manuscript use | Notes |
|---|---|---|
| `paper_framework_reports_protocol_v2_formal_formal_seed_results.csv` | Table 3, source-only / Model2 Balanced / Stage-1 / Stage-2 formal comparison | Main protocol-v2 three-seed summary. Source-only is interpreted as domain-shift evidence; Model2 Balanced is the fair target-domain adaptation baseline. |
| `paper_framework_reports_precision_first_ssod_formal_strict_ssod_formal_final_seed_results.csv` | Table 4 and strict SSOD results in Results section | Precision-first strict SSOD formal seed results with calibrated external evaluation. |

## Third-Domain Evaluation

| File | Manuscript use | Notes |
|---|---|---|
| `paper_framework_reports_third_domain_outdoor_zenodo_formal_third_domain_zero_shot_seed_results.csv` | Table 5 seed-level third-domain image-level evaluation | Outdoor oil palm dataset was not used in training. This is image-level, not detection mAP. |
| `paper_framework_reports_third_domain_outdoor_zenodo_formal_third_domain_zero_shot_summary.csv` | Table 5 mean and standard deviation values | Summarizes the third-domain image-level zero-shot comparison. |

## Under-Ripe Error Analysis

| File | Manuscript use | Notes |
|---|---|---|
| `paper_framework_reports_under_ripe_analysis_model2_balanced_seed2026_locked_external_classwise_ap.csv` | Table 6 and under-ripe limitation discussion | Class-wise AP for Model2 Balanced on locked external evaluation. |
| `paper_framework_reports_under_ripe_analysis_model2_balanced_seed2026_under_ripe_analysis_summary.json` | Table 6 false-negative and misclassification counts | Seed-2026 under-ripe error summary for Model2 Balanced. |
| `paper_framework_reports_under_ripe_analysis_strict_ssod_seed2026_locked_external_classwise_ap.csv` | Table 6 and strict SSOD under-ripe comparison | Class-wise AP for strict SSOD on locked external evaluation. |
| `paper_framework_reports_under_ripe_analysis_strict_ssod_seed2026_under_ripe_analysis_summary.json` | Table 6 false-negative and misclassification counts | Seed-2026 under-ripe error summary for strict SSOD. |

## Interpretation Rules

- Do not interpret the source-only external result as the main method baseline; it is used to demonstrate domain shift.
- Use Model2 Balanced as the primary fair supervised/adaptation baseline.
- Interpret strict SSOD as a conservative high-precision operating point.
- Treat third-domain results as exploratory because the third-domain protocol is image-level and smaller than the locked external detection test.
- The external preprocessed FFB protocol masks unavailable `empty` and `overripe` categories.
