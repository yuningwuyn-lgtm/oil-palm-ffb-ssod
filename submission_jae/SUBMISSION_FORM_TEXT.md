# JAE Submission Form Text

Use this file when filling the Journal of Agricultural Engineering online submission form.

## Manuscript Title

A Quality-Controlled Semi-Supervised Detection Framework for Cross-Domain Oil Palm Fresh Fruit Bunch Maturity Grading

## Article Type

Original Article

## Author

WU YUNING

## Affiliation

School of Electrical Engineering and Artificial Intelligence, Xiamen University Malaysia, Jalan Sunsuria, Bandar Sunsuria, 43900 Sepang, Selangor Darul Ehsan, Malaysia

## Corresponding Author Email

eee2309312@xmu.edu.my

## Abstract

Automated oil palm fresh fruit bunch (FFB) maturity grading can support harvesting decisions, reduce manual subjectivity, and improve traceability in plantation operations. However, visual detection models trained on one image source often degrade when deployed across plantations, acquisition devices, lighting conditions, and dataset annotation protocols. This study presents a quality-controlled semi-supervised object detection framework for cross-domain oil palm FFB maturity grading. The framework starts from a YOLOv8n detector trained on a labeled source-domain video-frame dataset, introduces class-space harmonization across heterogeneous datasets, applies scene-disjoint evaluation to reduce frame-level leakage, and evaluates external-domain performance on a separate preprocessed FFB dataset with a four-class masked protocol. To improve robustness, the proposed pipeline combines class-balanced target-domain adaptation with strict pseudo-label filtering based on confidence, bounding-box geometry, edge-contact penalties, folder-level weak-label consistency, multi-view augmentation consistency, and class-wise dynamic confidence thresholds. Three random seeds were used for the main formal comparison. The source-only baseline achieved high scene-disjoint source-domain performance but collapsed on the locked external domain, with external mAP@0.5 of 0.0372 and mAP@0.5:0.95 of 0.0130, confirming severe domain shift. A class-balanced target-domain adaptation model improved external mAP@0.5 to 0.6973 and mAP@0.5:0.95 to 0.4758. A precision-first strict SSOD model achieved calibrated external mAP@0.5 of 0.7240, mAP@0.5:0.95 of 0.4863, precision of 0.8753, recall of 0.5364, and F1-score of 0.6321. A third-domain zero-shot image-level evaluation further indicated improved transfer relative to the source-only baseline. The results show that external-domain adaptation and quality-controlled pseudo-label selection can produce a more conservative and field-oriented FFB maturity detector, although under-ripe fruit bunches remain the main error source.

## Keywords

agricultural engineering; domain adaptation; fresh fruit bunch; maturity grading; object detection; oil palm

## Availability of Data and Materials

The source and external datasets used in this study were derived from publicly available or locally prepared oil palm FFB datasets. The manuscript repository contains the class-space definition, selected formal evaluation outputs, and the code needed to reproduce the preparation and evaluation protocol. Raw images and trained weights are not redistributed in the repository because their redistribution depends on the licenses and access conditions of the original datasets. Processed split manifests and additional evaluation outputs can be shared by the corresponding author upon reasonable request where permitted by the original data licenses.

## Code Availability

The implementation was developed in Python using the Ultralytics YOLOv8 API. The pipeline includes dataset preparation, scene-disjoint splitting, duplicate checking, pseudo-label generation, quality scoring, class-wise threshold calibration, and structured JSON/CSV reporting. The public repository is available at https://github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod. The repository contains source code, configuration templates, manuscript files, and selected evaluation reports; raw third-party datasets and trained weights are excluded because their redistribution depends on the licenses and access conditions of the original data providers.

## Competing Interests

The author declares no conflict of interest.

## Funding

Not applicable. This research received no external funding.

## Acknowledgments

The author thanks the maintainers of the public oil palm FFB datasets and the open-source computer vision libraries used in this work.

## Authors' Contributions

WU YUNING: conceptualization, methodology, software, validation, formal analysis, data curation, writing-original draft preparation, visualization, and project administration. The contribution statement follows the CRediT taxonomy.

## Declaration of Generative Artificial Intelligence and Artificial Intelligence-Assisted Technologies in the Writing Process

During the preparation of this work, the author used OpenAI ChatGPT/Codex to assist with language polishing, code organization, validation-checklist preparation, and manuscript formatting. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

## Supporting Agencies

No external supporting agency was involved in this study.
