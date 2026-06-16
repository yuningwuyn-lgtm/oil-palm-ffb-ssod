"""
Build a compact EndNote import pack for the oil-palm cross-domain SSOD paper.

The pack contains:
- oilpalm_core_references.ris for EndNote import
- recommended_reading.csv with paper roles and reading priority
- openly available PDFs where a stable public URL exists
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

import requests


OUTPUT_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\literature\oilpalm")
PDF_ROOT = OUTPUT_ROOT / "pdfs"


PAPERS = [
    {
        "priority": "A",
        "category": "FFB domain",
        "title": "Annotated Datasets of Oil Palm Fruit Bunch Piles for Ripeness Grading Using Deep Learning",
        "authors": ["Suharjito", "Franz Adeta Junior", "Yosua Putra Koeswandy", "Debi", "Pratiwi Wahyu Nurhayati", "Muhammad Asrol", "Marimin"],
        "year": "2023",
        "type": "JOUR",
        "journal": "Scientific Data",
        "volume": "10",
        "pages": "72",
        "doi": "10.1038/s41597-023-01958-x",
        "url": "https://www.nature.com/articles/s41597-023-01958-x",
        "pdf": "https://www.nature.com/articles/s41597-023-01958-x.pdf",
        "note": "Primary source-domain dataset paper. Use it to explain the six FFB classes and why video/scene-disjoint evaluation is necessary.",
    },
    {
        "priority": "A",
        "category": "FFB domain",
        "title": "Outdoor oil palm fruit ripeness dataset",
        "authors": ["Munirah Rosbi", "Nurul Hidayah Ibrahim", "Noor Hasmiza Harun", "Mohd Aizuddin Abd Rahman"],
        "year": "2024",
        "type": "JOUR",
        "journal": "Data in Brief",
        "volume": "55",
        "pages": "110667",
        "doi": "10.1016/j.dib.2024.110667",
        "url": "https://www.sciencedirect.com/science/article/pii/S2352340924006346",
        "pdf": "https://ars.els-cdn.com/content/image/1-s2.0-S2352340924006346-main.pdf",
        "note": "Strict unseen third-domain dataset used only for zero-shot image-level evaluation in the current pipeline.",
    },
    {
        "priority": "A",
        "category": "Agricultural domain adaptation",
        "title": "Easy domain adaptation method for filling the species gap in deep learning-based fruit detection",
        "authors": ["Wenli Zhang", "Kang Chen", "Jianhua Wang"],
        "year": "2021",
        "type": "JOUR",
        "journal": "Horticulture Research",
        "volume": "8",
        "pages": "119",
        "doi": "10.1038/s41438-021-00553-8",
        "url": "https://academic.oup.com/hr/article/doi/10.1038/s41438-021-00553-8/6446655",
        "pdf": "https://academic.oup.com/hr/article-pdf/doi/10.1038/s41438-021-00553-8/43418251/41438_2021_article_553.pdf",
        "note": "Closest agricultural precedent for pseudo-label self-learning under domain shift. Compare your weak-label-guided quality filtering against its image translation and self-learning design.",
    },
    {
        "priority": "A",
        "category": "Agricultural domain adaptation",
        "title": "EasyDAM_V2: Efficient Data Labeling Method for Multishape, Cross-Species Fruit Detection",
        "authors": ["Wenli Zhang", "Kang Chen", "Jianhua Wang"],
        "year": "2022",
        "type": "JOUR",
        "journal": "Plant Phenomics",
        "volume": "2022",
        "pages": "9761674",
        "doi": "10.34133/2022/9761674",
        "url": "https://spj.science.org/doi/10.34133/2022/9761674",
        "pdf": "https://pdfs.semanticscholar.org/8855/ef9c3e8d539b353462530df523725b016df7.pdf",
        "note": "Important agricultural comparison for dynamic pseudo-label thresholds and cross-species fruit detection.",
    },
    {
        "priority": "A",
        "category": "Agricultural domain adaptation",
        "title": "Domain adaptive fruit detection method based on multiple alignments",
        "authors": ["An Guo", "Kaiqiong Sun", "Meng Wang"],
        "year": "2023",
        "type": "JOUR",
        "journal": "Journal of Intelligent & Fuzzy Systems",
        "volume": "45",
        "issue": "4",
        "doi": "10.3233/JIFS-232104",
        "url": "https://journals.sagepub.com/doi/abs/10.3233/JIFS-232104",
        "note": "Direct agricultural DA comparison: image alignment, feature alignment, YOLOv5, knowledge distillation, and mean teacher.",
    },
    {
        "priority": "A",
        "category": "Agricultural domain adaptation",
        "title": "DODA: Adapting Object Detectors to Dynamic Agricultural Environments in Real-Time with Diffusion",
        "authors": ["Shuai Xiang", "Pieter M. Blok", "James Burridge", "Haozhou Wang", "Wei Guo"],
        "year": "2026",
        "type": "CONF",
        "journal": "Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision",
        "pages": "4797-4807",
        "url": "https://openaccess.thecvf.com/content/WACV2026/html/Xiang_DODA_Adapting_Object_Detectors_to_Dynamic_Agricultural_Environments_in_Real-Time_WACV_2026_paper.html",
        "pdf": "https://openaccess.thecvf.com/content/WACV2026/papers/Xiang_DODA_Adapting_Object_Detectors_to_Dynamic_Agricultural_Environments_in_Real-Time_WACV_2026_paper.pdf",
        "note": "Recent agricultural DA reference. Use in discussion to contrast fast diffusion-based adaptation with your quality-controlled weak-label SSOD route.",
    },
    {
        "priority": "A",
        "category": "DAOD method",
        "title": "Cross-Domain Adaptive Teacher for Object Detection",
        "authors": ["Yu-Jhe Li", "Xiaoliang Dai", "Chih-Yao Ma", "Yen-Cheng Liu", "Kan Chen", "Bichen Wu", "Zijian He", "Kris Kitani", "Peter Vajda"],
        "year": "2022",
        "type": "CONF",
        "journal": "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "pages": "7581-7590",
        "doi": "10.1109/CVPR52688.2022.00743",
        "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.html",
        "pdf": "https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.pdf",
        "note": "Canonical DAOD teacher-student baseline: EMA teacher, adversarial alignment, and weak-strong augmentation. Cite when explaining why source-only pseudo-labeling can fail.",
    },
    {
        "priority": "A",
        "category": "DAOD method",
        "title": "SSDA-YOLO: Semi-supervised domain adaptive YOLO for cross-domain object detection",
        "authors": ["Jianing Wang", "Tian Han", "Jianxiong Yin", "Guang Yang", "Xiaoyang Wang"],
        "year": "2023",
        "type": "JOUR",
        "journal": "Computer Vision and Image Understanding",
        "volume": "229",
        "pages": "103649",
        "doi": "10.1016/j.cviu.2023.103649",
        "url": "https://www.sciencedirect.com/science/article/pii/S1077314223000292",
        "pdf": "https://arxiv.org/pdf/2211.02213.pdf",
        "note": "Essential method comparison because it combines one-stage YOLO with semi-supervised domain adaptation.",
    },
    {
        "priority": "B",
        "category": "DAOD method",
        "title": "Masked Retraining Teacher-Student Framework for Domain Adaptive Object Detection",
        "authors": ["Zijing Zhao", "Sitong Wei", "Qingchao Chen", "Dehui Li", "Yifan Yang", "Yuxin Peng", "Yang Liu"],
        "year": "2023",
        "type": "CONF",
        "journal": "Proceedings of the IEEE/CVF International Conference on Computer Vision",
        "pages": "19039-19049",
        "url": "https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection_ICCV_2023_paper.html",
        "pdf": "https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection_ICCV_2023_paper.pdf",
        "note": "Useful comparison for masked retraining and selective teacher-student adaptation.",
    },
    {
        "priority": "B",
        "category": "SSOD method",
        "title": "Rethinking Pseudo Labels for Semi-supervised Object Detection",
        "authors": ["Hengduo Li", "Zuxuan Wu", "Abhinav Shrivastava", "Larry S. Davis"],
        "year": "2022",
        "type": "CONF",
        "journal": "Proceedings of the AAAI Conference on Artificial Intelligence",
        "volume": "36",
        "issue": "2",
        "pages": "1314-1322",
        "doi": "10.1609/aaai.v36i2.20019",
        "url": "https://ojs.aaai.org/index.php/AAAI/article/view/20019",
        "pdf": "https://ojs.aaai.org/index.php/AAAI/article/download/20019/19778",
        "note": "Method justification for localization-aware pseudo-label quality, dynamic thresholds, and class imbalance handling.",
    },
    {
        "priority": "B",
        "category": "SSOD method",
        "title": "PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection",
        "authors": ["Hengduo Li", "Zuxuan Wu", "Hao Chen", "Xiujun Liang", "Larry S. Davis"],
        "year": "2022",
        "type": "CONF",
        "journal": "European Conference on Computer Vision",
        "url": "https://arxiv.org/abs/2203.16317",
        "pdf": "https://arxiv.org/pdf/2203.16317.pdf",
        "note": "Direct support for multiview consistency and pseudo-label filtering in your framework.",
    },
    {
        "priority": "B",
        "category": "UDA method",
        "title": "MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation",
        "authors": ["Lukas Hoyer", "Dengxin Dai", "Haoran Wang", "Luc Van Gool"],
        "year": "2023",
        "type": "CONF",
        "journal": "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "url": "https://openaccess.thecvf.com/content/CVPR2023/html/Hoyer_MIC_Masked_Image_Consistency_for_Context-Enhanced_Domain_Adaptation_CVPR_2023_paper.html",
        "pdf": "https://openaccess.thecvf.com/content/CVPR2023/papers/Hoyer_MIC_Masked_Image_Consistency_for_Context-Enhanced_Domain_Adaptation_CVPR_2023_paper.pdf",
        "note": "Masked target-image consistency reference. Your exploratory MIC-style proxy should be described as a proxy, not an exact reproduction.",
    },
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:90]


def download_pdf(url: str, output_path: Path) -> Optional[Path]:
    if output_path.exists() and output_path.stat().st_size > 10_000:
        return output_path
    try:
        response = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        if len(response.content) < 10_000 or b"%PDF" not in response.content[:1024]:
            raise ValueError("Response is not a valid PDF.")
        output_path.write_bytes(response.content)
        print(f"Downloaded PDF: {output_path.name}")
        return output_path
    except Exception as exc:
        print(f"WARNING: PDF download failed: {url}: {exc}")
        return None


def ris_entry(paper: dict, attachment: Optional[Path]) -> str:
    lines = [f"TY  - {paper['type']}"]
    lines.extend(f"AU  - {author}" for author in paper["authors"])
    lines.extend(
        [
            f"TI  - {paper['title']}",
            f"PY  - {paper['year']}",
            f"JO  - {paper.get('journal', '')}",
        ]
    )
    for ris_key, field in [("VL", "volume"), ("IS", "issue"), ("SP", "pages"), ("DO", "doi"), ("UR", "url")]:
        if paper.get(field):
            lines.append(f"{ris_key}  - {paper[field]}")
    lines.append(f"KW  - oilpalm")
    lines.append(f"KW  - {paper['category']}")
    lines.append(f"N1  - Reading priority {paper['priority']}. {paper['note']}")
    if attachment:
        lines.append(f"L1  - {attachment.resolve()}")
    lines.append("ER  - ")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PDF_ROOT.mkdir(parents=True, exist_ok=True)
    reading_rows = []
    entries = []
    for index, paper in enumerate(PAPERS, start=1):
        attachment = None
        if paper.get("pdf"):
            attachment = download_pdf(paper["pdf"], PDF_ROOT / f"{index:02d}_{slug(paper['title'])}.pdf")
        entries.append(ris_entry(paper, attachment))
        reading_rows.append(
            {
                "priority": paper["priority"],
                "category": paper["category"],
                "title": paper["title"],
                "year": paper["year"],
                "doi": paper.get("doi", ""),
                "url": paper["url"],
                "pdf_downloaded": str(bool(attachment)).lower(),
                "local_pdf": str(attachment or ""),
                "why_read": paper["note"],
            }
        )

    ris_path = OUTPUT_ROOT / "oilpalm_core_references.ris"
    ris_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8", newline="\n")
    csv_path = OUTPUT_ROOT / "recommended_reading.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(reading_rows[0]))
        writer.writeheader()
        writer.writerows(reading_rows)
    print(f"RIS: {ris_path}")
    print(f"Reading list: {csv_path}")
    print(f"Records: {len(PAPERS)}")


if __name__ == "__main__":
    main()

