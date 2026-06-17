"""Create a clean local JAE submission bundle.

The bundle contains only files useful during online submission or editorial
review. It deliberately excludes datasets, trained weights, generated training
runs, cache files, and local authentication artifacts.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1.0.9-jae-declaration-spelling"
BUNDLE_NAME = f"jae_submission_bundle_{VERSION}"
DIST = ROOT / "dist"
BUNDLE_DIR = DIST / BUNDLE_NAME
ZIP_PATH = DIST / f"{BUNDLE_NAME}.zip"


FILES = [
    ("manuscript_jae/main.pdf", "01_manuscript/main.pdf"),
    ("submission_jae/cover_letter.md", "02_submission_text/cover_letter.md"),
    ("submission_jae/editor_comments.md", "02_submission_text/editor_comments.md"),
    ("submission_jae/SUBMISSION_FORM_TEXT.md", "02_submission_text/SUBMISSION_FORM_TEXT.md"),
    ("submission_jae/suggested_reviewers_template.csv", "02_submission_text/suggested_reviewers_template.csv"),
    ("submission_jae/ARTIFACT_CHECKSUMS.md", "03_quality_control/ARTIFACT_CHECKSUMS.md"),
    ("submission_jae/FINAL_SUBMISSION_PACKAGE.md", "03_quality_control/FINAL_SUBMISSION_PACKAGE.md"),
    ("submission_jae/submission_checklist.md", "03_quality_control/submission_checklist.md"),
    ("submission_jae/pdf_visual_audit.md", "03_quality_control/pdf_visual_audit.md"),
    ("FINAL_READINESS_AUDIT.md", "03_quality_control/FINAL_READINESS_AUDIT.md"),
    ("RELEASE_NOTES_JAE_SUBMISSION.md", "03_quality_control/RELEASE_NOTES_JAE_SUBMISSION.md"),
    ("REPRODUCIBILITY.md", "04_reproducibility/REPRODUCIBILITY.md"),
    ("reports/REPORT_MANIFEST.md", "04_reproducibility/REPORT_MANIFEST.md"),
    ("CITATION.cff", "04_reproducibility/CITATION.cff"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def copy_files() -> list[tuple[str, int, str]]:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int, str]] = []
    for source, target in FILES:
        src = ROOT / source
        if not src.exists():
            raise FileNotFoundError(f"Missing bundle source: {source}")
        dst = BUNDLE_DIR / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append((target, dst.stat().st_size, sha256(dst)))
    return rows


def write_manifest(rows: list[tuple[str, int, str]]) -> None:
    lines = [
        "# JAE Submission Bundle Manifest",
        "",
        f"Version: `{VERSION}`",
        "",
        "Use `01_manuscript/main.pdf` as the main manuscript PDF upload.",
        "Copy text from `02_submission_text/` into the online submission form.",
        "Files under `03_quality_control/` and `04_reproducibility/` are support records and should be uploaded only if the journal system requests them.",
        "",
        "This bundle excludes raw datasets, trained weights, prepared dataset copies, local paths, authentication files, and training run folders.",
        "",
        "| File | Size bytes | SHA256 |",
        "|---|---:|---|",
    ]
    for rel, size, digest in rows:
        lines.append(f"| `{rel}` | {size} | `{digest}` |")
    lines.append("")
    (BUNDLE_DIR / "BUNDLE_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def write_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUNDLE_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(DIST).as_posix())


def main() -> None:
    rows = copy_files()
    write_manifest(rows)
    write_zip()
    print(f"Bundle directory: {BUNDLE_DIR}")
    print(f"Bundle zip: {ZIP_PATH}")
    print(f"Zip SHA256: {sha256(ZIP_PATH)}")


if __name__ == "__main__":
    main()
