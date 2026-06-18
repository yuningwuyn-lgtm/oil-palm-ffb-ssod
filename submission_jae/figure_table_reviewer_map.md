# Figure and Table Reviewer Map

This map records why each table or figure is included in the manuscript and which reviewer question it is intended to answer. It is a quality-control artifact for submission preparation, not a replacement for the manuscript captions.

## Journal Placement Rule

The Journal of Agricultural Engineering guide requires tables and figures to be placed at the end of the manuscript, while also requiring each table and figure to be cited in the text and supplied with relevant captions. The manuscript follows this pattern: the body text cites each artifact where it supports the argument, and the tables and figures are collected at the end of `manuscript_jae/main.pdf`.

## Tables

| Artifact | First text use | Reviewer-facing purpose |
|---|---|---|
| Table 1. Dataset roles used in the cross-domain evaluation protocol | Materials and Methods, Datasets and Class Harmonization | Shows that the work separates source-domain data, external adaptation data, and third-domain exploratory evaluation instead of mixing evidence sources. |
| Table 2. Unified class scheme and external-domain masked evaluation setting | Materials and Methods, Datasets and Class Harmonization | Explains the six-class harmonization and why external evaluation is masked to four classes when empty and overripe are absent. |
| Table 3. Formal three-seed detection results | Results, Domain Shift from Source-Only Training | Provides the main numerical evidence for source-domain performance, external collapse, target-domain adaptation, and strict SSOD comparison. |
| Table 4. Precision-first strict SSOD calibrated external-domain results | Results, Strict SSOD and High-Precision Calibration | Shows the high-precision operating point and makes the precision-recall trade-off explicit. |
| Table 5. Third-domain zero-shot image-level evaluation | Results, Third-Domain Zero-Shot Image-Level Evaluation | Supports the exploratory claim that transfer improves on a separate outdoor domain, while not overstating detection-level generalization. |
| Table 6. Seed-2026 under-ripe class analysis on the locked external dataset | Results, Under-Ripe Error Analysis | Documents the main remaining failure mode and prevents the paper from hiding class-level weakness behind aggregate metrics. |

## Figures

| Artifact | First text use | Reviewer-facing purpose |
|---|---|---|
| Figure 1. Quality-controlled cross-domain SSOD workflow | Materials and Methods, Datasets and Class Harmonization | Gives the reviewer a one-page overview of data roles, split controls, model stages, pseudo-label quality control, and evaluation outputs. |
| Figure 2. Model-level flow of the proposed SSOD pipeline | Materials and Methods, Datasets and Class Harmonization | Clarifies how Model 0, Model 2 Balanced, and strict SSOD relate, preventing misinterpretation that the source-only collapse is the main performance claim. |
| Figure 3. External-domain model comparison plot | Results, Strict SSOD and High-Precision Calibration | Visualizes the main external-domain comparison and helps the reviewer see that the fair baseline is Model 2 Balanced, not only source-only YOLOv8n. |
| Figure 4. External-domain calibration curve | Results, Strict SSOD and High-Precision Calibration | Supports the claim that the proposed setting prioritizes precision and requires calibrated operating thresholds. |
| Figure 5. External-domain confusion matrix for Model 2 Balanced | Results, Under-Ripe Error Analysis | Shows class-level error structure for the main fair adaptation baseline. |
| Figure 6. Locked external-domain confusion matrix for strict SSOD | Results, Under-Ripe Error Analysis | Shows the class-level behavior after strict SSOD and supports the discussion of conservative filtering and under-ripe limitations. |

## Reviewer Interpretation Guardrails

- Model 0 is included as domain-shift evidence, not as the fair baseline for claiming the proposed method's gain.
- Model 2 Balanced is the main fair baseline because it uses the target-domain training portion under the harmonized class protocol.
- Strict SSOD is presented as a high-precision operating point; its contribution is conservative deployment behavior, not a large headline mAP increase.
- The third-domain evaluation is exploratory because it is image-level and smaller than the locked external object-detection test.
- Under-ripe remains the main limitation and is deliberately shown through both table-level and confusion-matrix evidence.
