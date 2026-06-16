# Final Readiness Audit for JAE Submission

This audit records the evidence that the repository and manuscript package are ready for Journal of Agricultural Engineering submission review.

## Journal Requirements Checked

Source: Journal of Agricultural Engineering submission page  
URL: `https://www.agroengineering.org/jae/about/submissions`

| JAE requirement | Current evidence | Status |
|---|---|---|
| Original Article manuscript should be submitted as one file | `manuscript_jae/main.pdf` is a single PDF containing text, tables, and figures | Pass |
| Structure should include Abstract, Introduction, Materials and Methods, Results, Conclusions, References | `manuscript_jae/main.tex` follows this structure | Pass |
| Text should be double-spaced and use 12-point font | `main.tex` uses `\documentclass[12pt]{article}` and `\doublespacing` | Pass |
| Abstract maximum is 400 words | Validator reports 254 words | Pass |
| References should not exceed 40 | Validator reports 15 references | Pass |
| Total number of tables and figures should not exceed 15 | Validator reports 6 tables and 4 figures, total 10 | Pass |
| Tables and figures should be placed at the end | Tables and figures are placed after references and author statements | Pass |
| References must be traceable and accessible | References include DOI or stable online source where available | Pass |
| Keywords should fit JAE guidance | Manuscript uses 6 alphabetically ordered keywords | Pass |
| Author-year citation style should be used | Manuscript uses `natbib` author-year citations | Pass |
| Compulsory declarations should be present | Availability, competing interest, funding, acknowledgements, author contribution, AI declaration, and supporting-agency statements are included | Pass |
| Suggest at least 3/4 potential reviewers in Comments to the Editor | `submission_jae/editor_comments.md` includes 4 reviewers with affiliation, email, expertise, reason, and conflict check | Pass |

## Scientific Framing Checked

| Risk | Mitigation in current manuscript | Status |
|---|---|---|
| The work may look like a generic YOLO benchmark | Manuscript frames the study as cross-domain oil palm FFB maturity grading for agricultural engineering decision support | Controlled |
| Source-only external result may appear artificially weak | Manuscript explicitly treats source-only external collapse as domain-shift evidence, not the main improvement baseline | Controlled |
| Target-domain adaptation could be mistaken for zero-shot generalization | Manuscript explicitly states Model2 Balanced uses external-domain training data and is not zero-shot | Controlled |
| SSOD gain may be overstated | Manuscript describes strict SSOD as a high-precision operating point with limited recall | Controlled |
| External dataset lacks some classes | Manuscript uses four-class masked evaluation and states `empty` and `overripe` are ignored for this protocol | Controlled |
| Third-domain evaluation may be overinterpreted | Manuscript states third-domain outdoor results are exploratory and image-level | Controlled |
| Under-ripe weakness may be hidden | Manuscript reports under-ripe as the main bottleneck and discusses practical causes | Controlled |
| Deployment claim may be too strong | Manuscript frames the system as human-in-the-loop decision support, not a full replacement for trained graders | Controlled |

## Repository Evidence

| Evidence | File or system | Status |
|---|---|---|
| Public repository | `https://github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod` | Pass |
| Frozen release | `v1.0.1-jae-author-guidelines` | Pass |
| GitHub release page | `https://github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod/releases/tag/v1.0.1-jae-author-guidelines` | Pass |
| Citation metadata | `CITATION.cff` | Pass |
| Release notes | `RELEASE_NOTES_JAE_SUBMISSION.md` | Pass |
| Reproducibility guide | `REPRODUCIBILITY.md` | Pass |
| Report manifest | `reports/REPORT_MANIFEST.md` | Pass |
| Release-readiness exclusions | `RELEASE_READINESS.md` | Pass |
| Submission file map | `submission_jae/FINAL_SUBMISSION_PACKAGE.md` | Pass |
| Online form text | `submission_jae/SUBMISSION_FORM_TEXT.md` | Pass |
| PDF checksum record | `submission_jae/ARTIFACT_CHECKSUMS.md` | Pass |
| PDF visual audit | `submission_jae/pdf_visual_audit.md` | Pass |
| Automated validator | `scripts/validate_submission_package.py` | Pass |
| GitHub Actions validation | `Submission package validation` passes on `main` | Pass |

## Validated Manuscript Artifact

| Item | Value |
|---|---|
| File | `manuscript_jae/main.pdf` |
| Pages | 20 |
| Size | 467092 bytes |
| SHA256 | `7F0580FFCC2460D84B7DF2EB9D811CC9D2579BFD8E777EFCDD8BB167E8B812D3` |
| Blank-like pages | 0 |
| Image pages | 3 |

## Visual PDF Audit

The final PDF was rendered through PyMuPDF for targeted visual inspection. Pages 1, 2, 6, 10, 13, and 16-20 were inspected, covering the title page, abstract/body text, declarations, references transition, all table pages, and all figure pages. No visible blank pages, text clipping, overlapping content, broken tables, or broken figure rendering were observed. The detailed record is stored in `submission_jae/pdf_visual_audit.md`.

## Automated Validation Summary

The validator confirms:

- Required manuscript and submission files exist.
- Abstract length is below 400 words.
- References are below 40.
- Tables plus figures are below 15.
- Citation keys and bibliography entries are consistent.
- JAE author-year citation style, keyword count/order, postal address, and generative AI declaration are checked.
- PDF checksum matches the recorded artifact checksum.
- PDF visual audit covers representative text, table, and figure pages.
- Reviewer list contains at least 3 reviewers and valid email fields.
- Submission checklist has no open items.
- Python source files compile.
- Known local/private path strings are absent.

## Remaining Non-Technical Risks

The following cannot be proven or eliminated by repository work:

- Editorial fit and reviewer preference.
- Whether reviewers agree that the experimental gain is sufficiently novel.
- Whether JAE requests Word format after initial PDF submission.
- Whether dataset-license interpretation requires additional wording from the original data providers.
- Whether the single-author submission raises institutional or authorship questions.

These risks are explicitly controlled as far as possible through conservative claims, public code, reproducibility materials, reviewer suggestions, and release-level validation.

## Final Assessment

The manuscript package is technically ready for JAE submission. The repository contains the manuscript, supporting code, selected formal reports, reproducibility guide, submission materials, validator, CI workflow, checksum record, and frozen release snapshot. The remaining risks are editorial/scientific judgement risks rather than missing-file, formatting, or reproducibility risks.
