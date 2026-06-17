# Comments to the Editor

Dear Editor,

I am submitting this manuscript as an Original Article for the Journal of Agricultural Engineering. The paper presents a quality-controlled semi-supervised YOLOv8 framework for cross-domain oil palm fresh fruit bunch maturity grading.

The work is positioned as an applied agricultural engineering study rather than a generic computer-vision benchmark. Its main focus is the evaluation and improvement of maturity detection under cross-domain conditions, using scene-disjoint source-domain evaluation, locked external-domain testing, four-class masked external evaluation, class-balanced target-domain adaptation, and conservative pseudo-label quality control.

The manuscript intentionally distinguishes three interpretations:

1. The source-only YOLOv8n result is used as evidence of domain shift, not as the main claimed improvement.
2. The Model2 Balanced result is treated as the primary target-domain adaptation baseline.
3. The strict SSOD model is reported as a high-precision operating point, with its recall limitation stated explicitly.

The repository linked in the manuscript contains the code, configuration templates, manuscript files, selected formal evaluation outputs, and report manifests:

https://github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod

The frozen submission-support snapshot is:

https://github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod/releases/tag/v1.0.5-jae-declaration-headings

Raw third-party datasets and trained weights are not redistributed because their reuse and redistribution depend on the original data providers' license and access conditions.

Suggested reviewers are provided below with names, affiliations, email addresses, expertise, and conflict checks.

Suggested reviewers:

1. Hafiz Rashidi Ramli  
   Affiliation: Universiti Putra Malaysia  
   Email: hrhr@upm.edu.my  
   Expertise: oil palm FFB ripeness detection, agricultural robotics, YOLO-based detection  
   Reason: Co-author of oil palm FFB ripeness detection review and YOLOv4 ripe FFB detection work; suitable for evaluating agricultural relevance and maturity-detection framing.  
   Conflict check: No known collaboration; cited related work only.

2. Anwar P. P. Abdul Majeed  
   Affiliation: Sunway University  
   Email: anwarm@sunway.edu.my  
   Expertise: applied artificial intelligence, oil palm fruit ripeness datasets, agricultural computer vision  
   Reason: Co-author of the outdoor oil palm fruit ripeness dataset; suitable for assessing dataset use and computer-vision methodology.  
   Conflict check: No known collaboration; dataset/paper author only.

3. Zaid Omar  
   Affiliation: Universiti Teknologi Malaysia  
   Email: zaidomar@utm.my  
   Expertise: oil palm fruit ripeness dataset, image-based maturity grading  
   Reason: Co-author of the outdoor oil palm fruit ripeness dataset; suitable for reviewing oil-palm dataset interpretation and maturity-class relevance.  
   Conflict check: No known collaboration; dataset/paper author only.

4. Preety Baglat  
   Affiliation: University of Madeira; Interactive Technologies Institute ITI-LARSyS  
   Email: preety.baglat@iti.larsys.pt  
   Expertise: YOLO-based agricultural harvest-readiness systems, field computer vision  
   Reason: Author of a recent Journal of Agricultural Engineering article on mobile YOLO banana harvest-readiness prediction; suitable for reviewing applied agricultural-engineering presentation and deployment framing.  
   Conflict check: No known collaboration; same target journal topic area only.
