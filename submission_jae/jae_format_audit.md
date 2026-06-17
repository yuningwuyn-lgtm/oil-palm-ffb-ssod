# Journal of Agricultural Engineering Format Audit

This file records the submission-oriented checks applied to the manuscript.

## Official Requirements Checked

Source: Journal of Agricultural Engineering submission page.

- Original Articles should be submitted as one file.
- Recommended structure: Abstract, Introduction, Materials and Methods, Results, Conclusions, References.
- The detailed guide also lists Discussion as part of the manuscript text.
- Abstract maximum: 400 words.
- References should not exceed 40.
- Total number of tables and figures should not exceed 15.
- Text should be double-spaced and use 12-point font.
- Text should include numbered lines.
- Keywords should be limited to the journal range and ordered consistently.
- The journal uses author-year citation style.
- References should be listed in strict alphabetical order by first author's last name.
- Tables and figures should be placed at the end of the document.
- Compulsory declarations use JAE-aligned headings for availability of data and materials, competing interests, funding, acknowledgments, authors' contributions, AI declaration, and supporting-agency statements.
- At least 3-4 potential reviewers should be suggested in the submission comments.

## Current Manuscript Status

- Abstract words: 254.
- References: 15.
- Tables: 6.
- Figures: 5.
- Total tables and figures: 11.
- Keywords: 6, alphabetically ordered.
- Citation style: author-year citations generated with `natbib`.
- Line numbering: enabled with the LaTeX `lineno` package.
- Discussion section: included as an independent section.
- Reference order: manually arranged by first author's last name.
- Reference display format: revised toward JAE examples using surname-initial author format, year after authors, abbreviated journal/proceedings titles, and traceable DOI or stable URL.
- Title-page address: full Xiamen University Malaysia postal address included.
- References traceability: DOI or stable online source added where available.
- Generative AI declaration: included in manuscript and online submission form text.
- Practical deployment framing: manuscript describes human-in-the-loop use at collection points, grading lines, or mobile inspection stations.
- Independent Discussion section: removed; engineering interpretation is included inside Results.
- Tables and figures: placed after References and author statements.
- Compile check: `pdflatex` completed successfully.
- Render check: manuscript PDF rendered to page images for visual inspection; no blank pages, missing figures, or overfull layout errors were detected.
- Automated validation: GitHub Actions `Submission package validation` passes on the public repository.

## Final Submission Decisions

- GitHub repository status confirmed as public.
- Dataset license wording is conservative: raw third-party datasets and trained weights are not redistributed.
- Suggested reviewers file includes 4 candidates with institutional email addresses.
- JAE accepts a single Word or PDF manuscript file; the prepared submission file is `manuscript_jae/main.pdf`.
