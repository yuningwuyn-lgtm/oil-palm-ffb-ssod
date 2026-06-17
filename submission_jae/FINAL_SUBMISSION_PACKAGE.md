# Final JAE Submission Package

Use this file as the submission checklist when uploading to the Journal of Agricultural Engineering online system.

## Files to Upload or Copy

| Submission step | File or text | Action |
|---|---|---|
| Manuscript file | `manuscript_jae/main.pdf` | Upload as the single manuscript PDF. |
| Cover letter | `submission_jae/cover_letter.md` | Copy into the cover-letter field or upload if the system provides a cover-letter file slot. |
| Comments to the Editor | `submission_jae/editor_comments.md` | Copy into the "Comments to the Editor" field. This file includes the suggested reviewers. |
| Submission form text | `submission_jae/SUBMISSION_FORM_TEXT.md` | Copy title, abstract, keywords, author details, availability statements, funding, and conflict statements into the online form. |
| Suggested reviewers | `submission_jae/suggested_reviewers_template.csv` | Use as backup if the system asks for reviewer information in separate fields. |
| Repository link | `https://github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod` | Add in the code availability / comments field if requested. |
| Frozen release link | `https://github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod/releases/tag/v1.0.9-jae-declaration-spelling` | Add if the system asks for a versioned software/archive link. |
| Artifact checksum | `submission_jae/ARTIFACT_CHECKSUMS.md` | Use locally to confirm that the uploaded manuscript PDF matches the validated version. |
| PDF visual audit | `submission_jae/pdf_visual_audit.md` | Local quality-control record; do not upload unless requested. |
| Frozen release notes | `RELEASE_NOTES_JAE_SUBMISSION.md` | Use to identify the exact repository snapshot submitted for review. |

## Optional Local Bundle

Generate a clean local submission bundle before uploading:

```powershell
python scripts/create_submission_bundle.py
```

The command writes `dist/jae_submission_bundle_v1.0.9-jae-declaration-spelling/` and a matching `.zip` archive. The bundle contains the manuscript PDF, cover letter, editor comments, online-form text, reviewer template, checksums, and reproducibility support files only. It excludes raw datasets, model weights, training runs, cache files, and local authentication artifacts.

## Main Manuscript Claims to Preserve

- The source-only YOLOv8n model is evidence of cross-domain shift, not the main improvement baseline.
- Model2 Balanced is the fair supervised target-domain adaptation baseline.
- Strict SSOD is a high-precision operating point with lower recall.
- The external preprocessed FFB evaluation is a four-class masked protocol.
- The third-domain outdoor evaluation is image-level and exploratory.
- Raw third-party datasets and trained weights are not redistributed.

## Last Verified Format Status

- Article type: Original Article.
- Manuscript file: one 20-page PDF containing full text, tables, and figures.
- Abstract: 254 words, unstructured.
- References: 15.
- Tables: 6.
- Figures: 5.
- Total tables and figures: 11.
- Tables and figures are placed at the end.
- PDF visual audit: targeted rendered-page inspection passed.
- Public code repository is available.
- GitHub Actions submission-package validation passes on `main`.
- Frozen release version: `v1.0.9-jae-declaration-spelling`.
- Suggested reviewers: 4.

## Manual Check During Submission

Before clicking final submit:

1. Confirm the author name appears as `WU YUNING`.
2. Confirm affiliation/address is entered as `School of Electrical Engineering and Artificial Intelligence, Xiamen University Malaysia, Jalan Sunsuria, Bandar Sunsuria, 43900 Sepang, Selangor Darul Ehsan, Malaysia`.
3. Confirm email is `eee2309312@xmu.edu.my`.
4. Confirm the manuscript category is `Original Article`.
5. Confirm no raw dataset or trained-weight file is uploaded.
6. Confirm the title, abstract, and keywords copied into the online form match `submission_jae/SUBMISSION_FORM_TEXT.md`.
7. Confirm the generative AI declaration is copied into the required declaration field if the online system asks for it separately.
