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

## Scientific Risk Controls

- [x] Source-only baseline is described as domain-shift evidence, not as the main claimed improvement.
- [x] Model2 Balanced is described as target-domain adaptation, not zero-shot generalization.
- [x] Strict SSOD is described as a high-precision operating point, not as a large mAP improvement.
- [x] External evaluation uses a four-class masked protocol because the external dataset lacks `empty` and `overripe`.
- [x] Third-domain results are explicitly described as image-level exploratory evaluation.
- [x] Under-ripe limitations are reported rather than hidden.

## Items to Confirm Before Final Submission

- [ ] Confirm whether the GitHub repository should be public before submission or only shared upon request.
- [ ] Confirm dataset license wording for all public and locally prepared datasets.
- [ ] Confirm whether Xiamen University Malaysia affiliation wording is correct.
- [ ] Confirm whether any institutional acknowledgement is required.
- [ ] Prepare 3-4 suggested reviewers with verified institutional email addresses.
- [ ] Convert the final manuscript to Word if the submission system requires `.docx` instead of PDF.
