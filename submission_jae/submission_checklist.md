# JAE Submission Checklist

This checklist is based on the Journal of Agricultural Engineering submission page and was prepared to reduce format-related return before peer review.

## Manuscript Format

- [x] Article type: Original Article.
- [x] One manuscript file prepared as PDF from `manuscript_jae/main.tex`.
- [x] Text uses 12-point font.
- [x] Text is double-spaced.
- [x] Abstract is unstructured.
- [x] Abstract is below 400 words.
- [x] Main structure follows: Abstract, Introduction, Materials and Methods, Results, Conclusions, References.
- [x] References are fewer than 40.
- [x] Total number of tables and figures is fewer than 15.
- [x] Tables and figures are placed at the end of the manuscript.
- [x] A report manifest is included under `reports/REPORT_MANIFEST.md` to map CSV/JSON outputs to manuscript tables and interpretations.
- [x] Editor comments are prepared under `submission_jae/editor_comments.md` for the online submission form.
- [x] Online submission form text is prepared under `submission_jae/SUBMISSION_FORM_TEXT.md`.
- [x] Final submission package map is prepared under `submission_jae/FINAL_SUBMISSION_PACKAGE.md`.
- [x] Manuscript PDF checksum is recorded under `submission_jae/ARTIFACT_CHECKSUMS.md`.

## Scientific Risk Controls

- [x] Source-only baseline is described as domain-shift evidence, not as the main claimed improvement.
- [x] Model2 Balanced is described as target-domain adaptation, not zero-shot generalization.
- [x] Strict SSOD is described as a high-precision operating point, not as a large mAP improvement.
- [x] External evaluation uses a four-class masked protocol because the external dataset lacks `empty` and `overripe`.
- [x] Third-domain results are explicitly described as image-level exploratory evaluation.
- [x] Under-ripe limitations are reported rather than hidden.

## Items to Confirm Before Final Submission

- [x] Confirm whether the GitHub repository should be public before submission or only shared upon request. Current status: public GitHub repository.
- [x] Confirm conservative dataset license wording for public and locally prepared datasets. Current wording: code and selected reports are shared; raw datasets and trained weights are not redistributed.
- [x] Confirm whether Xiamen University Malaysia affiliation wording is correct. Current manuscript wording: School of Electrical Engineering and Artificial Intelligence, Xiamen University Malaysia.
- [x] Confirm whether any institutional acknowledgement is required. Current manuscript wording: no external supporting agency was involved.
- [x] Prepare 3-4 suggested reviewers with verified institutional email addresses.
- [x] Confirm upload format. JAE accepts a single WORD or PDF file; the prepared submission file is `manuscript_jae/main.pdf`.
