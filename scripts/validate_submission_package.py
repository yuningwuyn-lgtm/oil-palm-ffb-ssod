"""Validate the JAE submission package without running model training.

The checks are intentionally lightweight so they can run locally and in GitHub
Actions. They verify manuscript structure, citation consistency, reviewer
metadata, selected repository hygiene, and Python syntax.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION_FAILED: {message}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_required_files() -> None:
    required = [
        "README.md",
        "DATASETS.md",
        "FINAL_READINESS_AUDIT.md",
        "REPRODUCIBILITY.md",
        "RELEASE_READINESS.md",
        "RELEASE_NOTES_JAE_SUBMISSION.md",
        "CITATION.cff",
        "LICENSE",
        "manuscript_jae/main.tex",
        "manuscript_jae/main.pdf",
        "submission_jae/cover_letter.md",
        "submission_jae/editor_comments.md",
        "submission_jae/SUBMISSION_FORM_TEXT.md",
        "submission_jae/FINAL_SUBMISSION_PACKAGE.md",
        "submission_jae/ARTIFACT_CHECKSUMS.md",
        "submission_jae/pdf_visual_audit.md",
        "submission_jae/submission_checklist.md",
        "submission_jae/suggested_reviewers_template.csv",
        "submission_jae/jae_format_audit.md",
        "reports/REPORT_MANIFEST.md",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    if missing:
        fail(f"Missing required files: {missing}")

    citation_text = read_text(ROOT / "CITATION.cff")
    release_text = read_text(ROOT / "RELEASE_NOTES_JAE_SUBMISSION.md")
    for text_name, text in [("CITATION.cff", citation_text), ("RELEASE_NOTES_JAE_SUBMISSION.md", release_text)]:
        if "1.0.5-jae-declaration-headings" not in text:
            fail(f"{text_name} does not include the frozen submission version")


def check_manuscript_tex() -> dict[str, object]:
    tex = read_text(ROOT / "manuscript_jae/main.tex")
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not abstract_match:
        fail("Abstract environment missing")

    abstract_words = len(re.findall(r"[A-Za-z0-9@.:-]+", abstract_match.group(1)))
    bibs = set(re.findall(r"\\bibitem(?:\[[^\]]+\])?\{([^}]+)\}", tex))
    cites: set[str] = set()
    for match in re.findall(r"\\cite[a-zA-Z*]*\{([^}]+)\}", tex):
        cites.update(part.strip() for part in match.split(","))

    tables = len(re.findall(r"\\begin\{table\}", tex))
    figures = len(re.findall(r"\\begin\{figure\}", tex))
    bibitem_labels = re.findall(r"\\bibitem\[([^\]]+)\]\{([^}]+)\}", tex)

    if abstract_words > 400:
        fail(f"Abstract exceeds 400 words: {abstract_words}")
    if len(bibs) > 40:
        fail(f"References exceed 40: {len(bibs)}")
    if tables + figures > 15:
        fail(f"Tables + figures exceed 15: {tables + figures}")
    if cites - bibs:
        fail(f"Missing bibliography entries: {sorted(cites - bibs)}")
    if bibs - cites:
        fail(f"Unused bibliography entries: {sorted(bibs - cites)}")
    if "\\cite{" in tex:
        fail("Numeric-style raw \\cite commands remain; use author-year \\citep or \\citet")
    if "human-in-the-loop" not in tex:
        fail("Applied human-in-the-loop deployment framing missing")
    if "github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod" not in tex:
        fail("Public repository URL missing from manuscript")
    if "\\section{Discussion}" not in tex:
        fail("JAE-style Discussion section missing")
    keyword_match = re.search(r"\\textbf\{Keywords:\}\s*(.+)", tex)
    if not keyword_match:
        fail("Keywords line missing")
    keywords = [item.strip() for item in keyword_match.group(1).split(";") if item.strip()]
    if not (3 <= len(keywords) <= 6):
        fail(f"JAE keywords should contain 3 to 6 items, found {len(keywords)}")
    if keywords != sorted(keywords, key=str.lower):
        fail(f"JAE keywords should be in alphabetical order: {keywords}")
    if "Declaration of Generative Artificial Intelligence" not in tex:
        fail("Required generative AI declaration section missing")
    for required_section in [
        "Availability of Data and Materials",
        "Competing Interests",
        "Funding",
        "Acknowledgements",
        "Authors' Contributions",
        "Supporting Agencies",
    ]:
        if required_section not in tex:
            fail(f"Required JAE declaration heading missing: {required_section}")
    if "Jalan Sunsuria, Bandar Sunsuria" not in tex:
        fail("Full institutional postal address missing from title page")
    if "\\usepackage{lineno}" not in tex or "\\linenumbers" not in tex:
        fail("JAE line numbering is missing; use lineno and enable \\linenumbers")
    reference_order = []
    for label, _key in bibitem_labels:
        author_part = label.split("(", 1)[0].strip()
        first_author = re.split(r"\s+and\s+|\s+et al\.", author_part, maxsplit=1)[0].strip()
        reference_order.append(first_author.lower())
    if reference_order != sorted(reference_order):
        fail(f"References are not in strict alphabetical order by first author: {reference_order}")

    return {
        "abstract_words": abstract_words,
        "references": len(bibs),
        "tables": tables,
        "figures": figures,
        "tables_plus_figures": tables + figures,
    }


def check_pdf() -> dict[str, object]:
    pdf_path = ROOT / "manuscript_jae/main.pdf"
    checksum_text = read_text(ROOT / "submission_jae/ARTIFACT_CHECKSUMS.md")
    pdf_bytes = pdf_path.read_bytes()
    digest = hashlib.sha256(pdf_bytes).hexdigest().upper()
    if digest not in checksum_text:
        fail("Manuscript PDF SHA256 does not match submission checksum file")
    if str(len(pdf_bytes)) not in checksum_text:
        fail("Manuscript PDF size does not match submission checksum file")
    try:
        import fitz  # type: ignore
    except Exception:
        return {
            "pdf_exists": pdf_path.exists(),
            "pdf_sha256": digest,
            "pdf_size_bytes": len(pdf_bytes),
            "pdf_render_check": "skipped_no_pymupdf",
        }

    doc = fitz.open(pdf_path)
    blank_pages: list[int] = []
    image_pages = 0
    for index, page in enumerate(doc):
        text = page.get_text().strip()
        images = page.get_images(full=True)
        drawings = page.get_drawings()
        if images:
            image_pages += 1
        if len(text) < 20 and not images and not drawings:
            blank_pages.append(index + 1)
    if blank_pages:
        fail(f"Blank-like pages detected: {blank_pages}")
    if len(doc) < 1:
        fail("PDF has no pages")
    if image_pages < 1:
        fail("PDF appears to have no figure/image pages")
    return {
        "pdf_pages": len(doc),
        "blank_like_pages": blank_pages,
        "image_pages": image_pages,
        "pdf_sha256": digest,
        "pdf_size_bytes": len(pdf_bytes),
    }


def check_submission_files() -> dict[str, object]:
    checklist = read_text(ROOT / "submission_jae/submission_checklist.md")
    open_items = [line for line in checklist.splitlines() if line.startswith("- [ ]")]
    if open_items:
        fail(f"Open checklist items remain: {open_items}")

    with (ROOT / "submission_jae/suggested_reviewers_template.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reviewers = list(csv.DictReader(handle))
    if len(reviewers) < 3:
        fail(f"At least 3 suggested reviewers required, found {len(reviewers)}")
    missing_email = [row.get("name", "") for row in reviewers if "@" not in row.get("email", "")]
    if missing_email:
        fail(f"Reviewer email missing or invalid for: {missing_email}")

    final_package = read_text(ROOT / "submission_jae/FINAL_SUBMISSION_PACKAGE.md")
    for token in [
        "manuscript_jae/main.pdf",
        "cover_letter.md",
        "editor_comments.md",
        "SUBMISSION_FORM_TEXT.md",
        "ARTIFACT_CHECKSUMS.md",
        "v1.0.5-jae-declaration-headings",
    ]:
        if token not in final_package:
            fail(f"Final package map missing token: {token}")
    cover_letter = read_text(ROOT / "submission_jae/cover_letter.md")
    editor_comments = read_text(ROOT / "submission_jae/editor_comments.md")
    for text_name, text in [("cover_letter.md", cover_letter), ("editor_comments.md", editor_comments)]:
        if "https://github.com/yuningwuyn-lgtm/oil-palm-ffb-ssod" not in text:
            fail(f"{text_name} missing public repository URL")
        if "v1.0.5-jae-declaration-headings" not in text:
            fail(f"{text_name} missing frozen release URL")

    form_text = read_text(ROOT / "submission_jae/SUBMISSION_FORM_TEXT.md")
    tex = read_text(ROOT / "manuscript_jae/main.tex")
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not abstract_match:
        fail("Cannot compare submission form abstract because manuscript abstract is missing")
    manuscript_abstract = " ".join(abstract_match.group(1).split())
    if manuscript_abstract not in " ".join(form_text.split()):
        fail("Submission form abstract does not match manuscript abstract")
    for phrase in [
        "A Quality-Controlled Semi-Supervised Detection Framework",
        "agricultural engineering; domain adaptation; fresh fruit bunch",
        "The author declares no conflict of interest.",
        "This research received no external funding.",
        "Availability of Data and Materials",
        "Competing Interests",
        "Authors' Contributions",
        "Declaration of Generative Artificial Intelligence",
    ]:
        if phrase not in form_text:
            fail(f"Submission form text missing phrase: {phrase}")

    return {"reviewer_count": len(reviewers), "open_checklist_items": len(open_items)}


def check_privacy_strings() -> None:
    patterns = [
        r"[A-Z]:\\",
        "Google" + "Dowload",
        "qaw" + "se",
        "jian" + "ding",
        "teacher" + "_report",
        "advisor" + "_review",
        "private " + "repository",
        "may be made " + "public",
        "TO" + "DO",
    ]
    checked_suffixes = {".md", ".tex", ".py", ".yaml", ".yml", ".csv", ".cff", ".txt"}
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if path == Path(__file__).resolve():
            continue
        if ".git" in path.parts or path.is_dir() or path.suffix.lower() not in checked_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if re.search(pattern, text):
                hits.append(f"{path.relative_to(ROOT)}:{pattern}")
    if hits:
        fail(f"Privacy or placeholder strings found: {hits[:20]}")


def check_python_compile() -> None:
    for path in sorted((ROOT / "src").glob("*.py")):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            fail(f"py_compile failed for {path.relative_to(ROOT)}: {result.stderr}")


def main() -> None:
    check_required_files()
    manuscript = check_manuscript_tex()
    pdf = check_pdf()
    submission = check_submission_files()
    check_privacy_strings()
    check_python_compile()
    print(
        json.dumps(
            {
                "status": "ok",
                "manuscript": manuscript,
                "pdf": pdf,
                "submission": submission,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
