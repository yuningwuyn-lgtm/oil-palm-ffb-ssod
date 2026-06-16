"""
Full semi-supervised object detection (SSOD) pipeline for oil palm Fresh Fruit
Bunch (FFB) maturity detection with Ultralytics YOLOv8.

Run examples:
    python full_ssod_ffb_pipeline.py --original-dataset "LOCAL_DATA_ROOT/OILPALM.yolov8/split_dataset" --new-images "LOCAL_DATA_ROOT/new_images"
    python full_ssod_ffb_pipeline.py --download-open-data --max-epochs 10 --self-training-iterations 1
    python full_ssod_ffb_pipeline.py --roboflow-zip-url "https://.../yolov8.zip"

Notes:
    - This is an end-to-end research pipeline, not a tiny demo. Zenodo's open
      outdoor FFB archive is about 1.2 GB, so downloading can take time.
    - Some public FFB datasets are image-level classification datasets rather
      than object detection datasets. When no boxes are available, this script
      can bootstrap YOLO detection labels using a full-image box per image.
      That is useful for a weak baseline, but real bounding-box annotations are
      preferred whenever available.
    - The SSOD blocks are inspired by consistency training and high-quality
      pseudo-label selection ideas used in object detection papers such as
      "Rethinking Pseudo Labels for Semi-Supervised Object Detection" and
      "PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised
      Object Detection".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import tarfile
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
import yaml
from PIL import Image, ImageDraw, ImageEnhance
from ultralytics import YOLO


# =============================================================================
# A. Canonical classes and open dataset sources
# =============================================================================

CANONICAL_CLASSES = {
    0: "abnormal",
    1: "empty",
    2: "overripe",
    3: "ripe",
    4: "under_ripe",
    5: "unripe",
}
CLASS_NAME_TO_ID = {name: class_id for class_id, name in CANONICAL_CLASSES.items()}
PREPROCESSED_FFB_EVAL_CLASS_IDS = (0, 3, 4, 5)
PREPROCESSED_FFB_EVAL_CLASSES = [CANONICAL_CLASSES[class_id] for class_id in PREPROCESSED_FFB_EVAL_CLASS_IDS]
DEFAULT_CALIBRATION_THRESHOLDS = "0.05,0.10,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90"
DEFAULT_FIXED_EVAL_THRESHOLDS = "0.25,0.40,0.50,0.60,0.70"
DEFAULT_CLASSWISE_THRESHOLD_GRID = "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"

def parse_class_float_map(text: str) -> Dict[int, float]:
    """
    Parse class-specific CLI overrides such as:
        abnormal=0.45,under_ripe=0.50

    Class-specific controls are useful for SCI-grade SSOD ablations because the
    hard minority classes should be tuned explicitly instead of hidden inside a
    single global pseudo-label threshold.
    """
    overrides: Dict[int, float] = {}
    if not text:
        return overrides
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected class=value override, got: {item}")
        name, value = item.split("=", 1)
        canonical = canonical_class_name(name)
        if canonical is None or canonical not in CLASS_NAME_TO_ID:
            raise ValueError(f"Unknown class in override: {name}")
        overrides[CLASS_NAME_TO_ID[canonical]] = float(value)
    return overrides


def parse_class_int_map(text: str) -> Dict[int, int]:
    """Parse class-specific integer controls such as abnormal=80,ripe=150."""
    return {class_id: int(value) for class_id, value in parse_class_float_map(text).items()}


def parse_class_list(text: str) -> Set[int]:
    class_ids: Set[int] = set()
    if not text:
        return class_ids
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        canonical = canonical_class_name(item)
        if canonical is None or canonical not in CLASS_NAME_TO_ID:
            raise ValueError(f"Unknown class in list: {item}")
        class_ids.add(CLASS_NAME_TO_ID[canonical])
    return class_ids


def parse_hard_class_source_map(text: str) -> Dict[int, Set[int]]:
    """
    Parse mappings such as:
        under_ripe:unripe|ripe,abnormal:empty|unripe|overripe
    """
    mapping: Dict[int, Set[int]] = {}
    if not text:
        return mapping
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected target:source|source mapping, got: {item}")
        target_name, source_text = item.split(":", 1)
        target = canonical_class_name(target_name)
        if target is None or target not in CLASS_NAME_TO_ID:
            raise ValueError(f"Unknown hard-class target: {target_name}")
        sources: Set[int] = set()
        for source_name in source_text.split("|"):
            source = canonical_class_name(source_name.strip())
            if source is None or source not in CLASS_NAME_TO_ID:
                raise ValueError(f"Unknown hard-class source: {source_name}")
            sources.add(CLASS_NAME_TO_ID[source])
        mapping[CLASS_NAME_TO_ID[target]] = sources
    return mapping

CLASS_ALIASES = {
    "abnormal": "abnormal",
    "damaged": "abnormal",
    "damaged_bunch": "abnormal",
    "damaged bunch": "abnormal",
    "empty": "empty",
    "empty_bunch": "empty",
    "empty bunch": "empty",
    "janjang_kosong": "empty",
    "janjang kosong": "empty",
    "overripe": "overripe",
    "over_ripe": "overripe",
    "over-ripe": "overripe",
    "terlalu_masak": "overripe",
    "ripe": "ripe",
    "ripe_ffb": "ripe",
    "ripe ffb": "ripe",
    "fresh_fruit_bunch_ripe": "ripe",
    "masak": "ripe",
    "tbs_masak": "ripe",
    "tbs masak": "ripe",
    "under_ripe": "under_ripe",
    "under-ripe": "under_ripe",
    "underripe": "under_ripe",
    "under ripe": "under_ripe",
    "kurang_masak": "under_ripe",
    "kurang masak": "under_ripe",
    "unripe": "unripe",
    "unripe_ffb": "unripe",
    "unripe ffb": "unripe",
    "fresh_fruit_bunch_unripe": "unripe",
    "fresh_fruit_bunch": None,
    "ffb": None,
    "flower": None,
    "immature": "unripe",
    "partially_ripe": "under_ripe",
    "partially ripe": "under_ripe",
    "fully_ripe": "ripe",
    "fully ripe": "ripe",
    "decayed": "abnormal",
    "rotten": "abnormal",
    "belum_masak": "unripe",
    "tbs_mentah": "unripe",
    "tbs mentah": "unripe",
    "tbs_abnormal": "abnormal",
    "tbs abnormal": "abnormal",
    "terlalu_masak": "overripe",
    "terlalu masak": "overripe",
}

ZENODO_RECORD_ID = "11114885"
ZENODO_RECORD_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_DIRECT_URL = (
    "https://zenodo.org/records/11114885/files/"
    "MunirahRosbi%2FOutdoor-Tenera-Oil-Palm-Fruit-Image-v1.zip?download=1"
)

OPEN_DATASET_SOURCES = [
    {
        "name": "zenodo_outdoor_tenera_ffb",
        "kind": "zenodo",
        "record_id": "11114885",
        "doi": "10.5281/zenodo.11114885",
        "classes": ["damaged_bunch", "empty_bunch", "unripe", "ripe", "overripe"],
        "notes": "Open outdoor oil palm FFB ripeness data. Missing canonical class: under_ripe.",
    },
    {
        "name": "mendeley_ordinal_ripeness_424y96m6sw",
        "kind": "mendeley",
        "dataset_id": "424y96m6sw",
        "version": "1",
        "doi": "10.17632/424y96m6sw.1",
        "classes": ["immature", "partially_ripe", "fully_ripe", "overripe", "decayed"],
        "notes": "Image-level ordinal ripeness dataset; converted to weak full-image boxes if no detection labels exist.",
    },
    {
        "name": "science_data_bank_candidates",
        "kind": "url_candidates",
        "urls": [],
        "classes": [],
        "notes": "Placeholder for public Science Data Bank mirrors. Add direct archive URLs with --extra-dataset-url.",
    },
]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SAFE_DELETE_ROOT: Optional[Path] = None


# =============================================================================
# B. Data structures
# =============================================================================

@dataclass
class BoxPrediction:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    support: int = 1
    ensemble_support: int = 1
    quality_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class ImagePseudoResult:
    image_path: Path
    scene: str
    accepted_boxes: List[BoxPrediction]
    rejected_boxes: List[BoxPrediction]
    status: str
    reasons: List[str] = field(default_factory=list)


@dataclass
class StageMetrics:
    stage: str
    precision: float
    recall: float
    map50: float
    map5095: float
    model_path: str
    data_yaml: str


class JsonRunLogger:
    """Append structured events for reproducible SSOD experiments."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.events: List[dict] = []
        ensure_dir(output_path.parent)

    def log(self, event_type: str, **payload) -> None:
        event = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event_type,
            **json_safe(payload),
        }
        self.events.append(event)
        self.flush()

    def flush(self) -> None:
        self.output_path.write_text(json.dumps(self.events, indent=2, ensure_ascii=False), encoding="utf-8")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return json_safe(value.__dict__)
    return value


# =============================================================================
# C. General filesystem helpers
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_safe_delete_root(path: Path) -> None:
    global SAFE_DELETE_ROOT
    SAFE_DELETE_ROOT = path.resolve()


def assert_safe_to_clear(path: Path) -> None:
    """
    Prevent accidental destructive deletes when users change --output-root.

    All generated folders must live under SAFE_DELETE_ROOT. Refuse drive roots,
    home directories, and paths outside the configured output tree.
    """
    resolved = path.resolve()
    if resolved.anchor and str(resolved) == resolved.anchor:
        raise ValueError(f"Refusing to clear drive root: {resolved}")
    forbidden = {Path.home().resolve(), Path.cwd().resolve()}
    if resolved in forbidden:
        raise ValueError(f"Refusing to clear protected directory: {resolved}")
    if SAFE_DELETE_ROOT is not None:
        try:
            resolved.relative_to(SAFE_DELETE_ROOT)
        except ValueError as exc:
            raise ValueError(f"Refusing to clear path outside output root: {resolved}") from exc


def clear_dir(path: Path) -> None:
    assert_safe_to_clear(path)
    if path.exists():
        for attempt in range(5):
            try:
                shutil.rmtree(path)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.5)
                for child in path.rglob("*"):
                    try:
                        os.chmod(child, 0o700)
                    except OSError:
                        pass
    ensure_dir(path)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def link_or_copy_file(src: Path, dst: Path) -> None:
    """Create a hardlink when possible, falling back to copy for cross-volume paths."""
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_train_cache(value: str):
    value = str(value).strip().lower()
    if value in {"", "none", "false", "0", "off"}:
        return False
    if value in {"true", "1", "on", "ram"}:
        return True
    if value == "disk":
        return "disk"
    raise ValueError(f"Unsupported --train-cache value: {value}")


def normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_class_name(value: str) -> Optional[str]:
    normalized = normalize_name(value)
    return CLASS_ALIASES.get(normalized, normalized if normalized in CLASS_NAME_TO_ID else None)


def safe_class_id(value: str | int) -> Optional[int]:
    if isinstance(value, int):
        return value if value in CANONICAL_CLASSES else None
    class_name = canonical_class_name(value)
    if class_name is None:
        return None
    return CLASS_NAME_TO_ID[class_name]


def infer_scene(image_path: Path, image_root: Path) -> str:
    try:
        relative = image_path.resolve().relative_to(image_root.resolve())
    except ValueError:
        return "root_scene"
    if len(relative.parts) <= 1:
        return "root_scene"
    return normalize_name(relative.parts[0])


def expected_class_from_folder(image_path: Path, image_root: Path) -> Optional[int]:
    try:
        relative = image_path.resolve().relative_to(image_root.resolve())
    except ValueError:
        relative = image_path
    if len(relative.parts) < 2:
        return None
    return safe_class_id(relative.parts[-2])


def image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


# =============================================================================
# D. Download and dataset conversion
# =============================================================================

def download_and_extract(url: str, dest: Path, filename: Optional[str] = None, timeout: int = 60) -> Path:
    """
    Download a zip/tar archive and extract it into dest.

    This function works with direct Zenodo, Roboflow export, Kaggle mirror, or
    any normal archive URL. For Zenodo records, prefer the API helper below.
    """
    ensure_dir(dest)
    filename = filename or url.split("?")[0].rstrip("/").split("/")[-1] or "downloaded_dataset.zip"
    archive_path = dest / filename
    extract_dir = dest / archive_path.stem

    if not archive_path.exists():
        print(f"Downloading: {url}")
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with archive_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:5.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")
            print()
    else:
        print(f"Archive already exists: {archive_path}")

    if extract_dir.exists() and list(extract_dir.iterdir()):
        print(f"Archive already extracted: {extract_dir}")
        return extract_dir

    ensure_dir(extract_dir)
    print(f"Extracting: {archive_path}")
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    return extract_dir


def download_zenodo_record(record_id: str, dest: Path) -> List[Path]:
    """
    Download all files from a Zenodo record using the public REST API.

    The outdoor FFB dataset at DOI 10.5281/zenodo.11114885 is exposed as
    record 11114885 and currently has one 1.2 GB zip file.
    """
    ensure_dir(dest)
    print(f"Fetching Zenodo record metadata: {record_id}")
    try:
        response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
        response.raise_for_status()
        record = response.json()
        files = record.get("files", [])
        if not files:
            raise ValueError("Zenodo record has no files.")
        extracted_dirs = []
        for file_info in files:
            links = file_info.get("links", {})
            url = links.get("self") or links.get("download")
            key = file_info.get("key", "zenodo_file.zip").replace("/", "_")
            if not url:
                continue
            extracted_dirs.append(download_and_extract(url, dest, filename=key))
        return extracted_dirs
    except Exception as exc:
        print(f"WARNING: Zenodo API download failed: {exc}")
        print("Falling back to known direct Zenodo file URL.")
        return [download_and_extract(ZENODO_DIRECT_URL, dest, filename="Outdoor-Tenera-Oil-Palm-Fruit-Image-v1.zip")]


def download_mendeley_dataset(dataset_id: str, version: str, dest: Path) -> List[Path]:
    """
    Try the public Mendeley Data file API, then gracefully skip if direct files
    are unavailable without browser authentication.
    """
    ensure_dir(dest)
    endpoints = [
        f"https://data.mendeley.com/public-files/datasets/{dataset_id}/files",
        f"https://data.mendeley.com/public-files/datasets/{dataset_id}/{version}/files",
    ]
    downloaded: List[Path] = []
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=60)
            if not response.ok:
                continue
            files = response.json()
            if isinstance(files, dict):
                files = files.get("files", [])
            for file_info in files:
                filename = file_info.get("filename") or file_info.get("name") or "mendeley_file.zip"
                file_id = file_info.get("id") or file_info.get("file_id")
                links = file_info.get("links", {})
                url = links.get("download") or links.get("self")
                if not url and file_id:
                    url = f"https://data.mendeley.com/public-files/datasets/{dataset_id}/files/{file_id}/download"
                if not url:
                    continue
                try:
                    downloaded.append(download_and_extract(url, dest, filename=filename))
                except Exception as exc:
                    print(f"WARNING: Could not download Mendeley file {filename}: {exc}")
            if downloaded:
                return downloaded
        except Exception as exc:
            print(f"WARNING: Mendeley endpoint failed {endpoint}: {exc}")
    print(f"WARNING: Mendeley dataset {dataset_id} was not downloaded. Open manually if needed: https://data.mendeley.com/datasets/{dataset_id}/{version}")
    return downloaded


def try_download_public_datasets(args: argparse.Namespace, dest: Path, logger: Optional[JsonRunLogger] = None) -> List[Tuple[str, Path]]:
    """
    Try multiple public oil-palm ripeness sources. Each source is optional:
    failed downloads are logged and skipped so local training remains runnable.
    """
    extracted: List[Tuple[str, Path]] = []
    for source in OPEN_DATASET_SOURCES:
        name = source["name"]
        try:
            if source["kind"] == "zenodo":
                for path in download_zenodo_record(source["record_id"], dest / name):
                    extracted.append((name, path))
            elif source["kind"] == "mendeley":
                for path in download_mendeley_dataset(source["dataset_id"], source.get("version", "1"), dest / name):
                    extracted.append((name, path))
            elif source["kind"] == "url_candidates":
                for idx, url in enumerate(source.get("urls", []), start=1):
                    extracted.append((name, download_and_extract(url, dest / name, filename=f"{name}_{idx}.zip")))
            if logger:
                logger.log("dataset_download_attempt", source=source, extracted=[p for _, p in extracted if _ == name])
        except Exception as exc:
            print(f"WARNING: Public dataset source failed: {name}: {exc}")
            if logger:
                logger.log("dataset_download_failed", source=source, error=str(exc))

    for idx, url in enumerate(args.extra_dataset_url or [], start=1):
        try:
            path = download_and_extract(url, dest / "extra_urls", filename=f"extra_dataset_{idx}.zip")
            extracted.append((f"extra_url_{idx}", path))
            if logger:
                logger.log("dataset_download_attempt", source={"name": f"extra_url_{idx}", "url": url}, extracted=[path])
        except Exception as exc:
            print(f"WARNING: Extra dataset URL failed: {url}: {exc}")
            if logger:
                logger.log("dataset_download_failed", source={"url": url}, error=str(exc))
    return extracted


def detect_yolo_dataset(data_dir: Path) -> Optional[Path]:
    for candidate in [data_dir / "data.yaml", data_dir / "data.yml", *data_dir.rglob("data.yaml")]:
        if candidate.exists():
            return candidate
    return None


def read_yolo_names(data_yaml: Path) -> Dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def remap_yolo_dataset(source_yaml: Path, output_root: Path) -> Path:
    """
    Copy an existing YOLO dataset and remap its classes into CANONICAL_CLASSES.
    """
    clear_dir(output_root)
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source_root = Path(data.get("path", source_yaml.parent))
    if not source_root.is_absolute():
        source_root = (source_yaml.parent / source_root).resolve()
    source_names = read_yolo_names(source_yaml)
    old_to_new = {}
    for old_id, name in source_names.items():
        new_id = safe_class_id(name)
        if new_id is not None:
            old_to_new[old_id] = new_id

    for split, yaml_key in [("train", "train"), ("valid", "val"), ("test", "test")]:
        raw_image_path = data.get(yaml_key)
        if raw_image_path is None:
            continue
        images_dir = Path(raw_image_path)
        if not images_dir.is_absolute():
            images_dir = source_root / images_dir
        if not images_dir.exists():
            # Some exported datasets keep data.yaml next to train/valid/test but
            # still write Roboflow-style "../train/images" paths.
            fallback = source_yaml.parent / split / "images"
            if fallback.exists():
                images_dir = fallback
        labels_dir = Path(str(images_dir).replace(f"{os.sep}images", f"{os.sep}labels"))
        dst_images = output_root / split / "images"
        dst_labels = output_root / split / "labels"
        ensure_dir(dst_images)
        ensure_dir(dst_labels)
        for image_path in list_images(images_dir):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            lines = []
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    old_id = int(float(parts[0]))
                except ValueError:
                    continue
                if old_id not in old_to_new:
                    continue
                lines.append(" ".join([str(old_to_new[old_id]), *parts[1:]]))
            if not lines:
                continue
            copy_file(image_path, dst_images / image_path.name)
            (dst_labels / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_harmonization_metadata(output_root, source_yaml.parent.name, list(source_names.values()))
    return write_data_yaml(output_root)


def resolve_yolo_images_dir(data_yaml: Path, split: str) -> Optional[Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    key = "val" if split == "valid" else split
    raw_path = data.get(key)
    if raw_path is None:
        return None
    source_root = Path(data.get("path", data_yaml.parent))
    if not source_root.is_absolute():
        source_root = (data_yaml.parent / source_root).resolve()
    images_dir = Path(raw_path)
    if not images_dir.is_absolute():
        images_dir = source_root / images_dir
    return images_dir


def source_group_from_image_name(image_name: str) -> str:
    """
    Estimate the acquisition scene/video from Roboflow-style frame names.

    A random image split leaks scene information when adjacent video frames land
    in train and test. Grouping by this coarse source name gives a stricter
    evaluation split for generalization claims.
    """
    stem = Path(image_name).stem.split(".rf.")[0]
    stem = re.sub(r"(?i)(?:frame|img|image|copy)[_-]?\d+$", "", stem)
    stem = re.sub(r"(?i)[_-]?(?:jpg|jpeg|png|bmp)$", "", stem)
    stem = re.sub(r"--\d+.*$", "", stem)
    stem = re.sub(r"\d+$", "", stem)
    return stem.strip("_- ") or Path(image_name).stem


def normalized_base_filename(image_path: Path) -> str:
    """
    Normalize filenames for duplicate/leakage checks across exported datasets.

    Roboflow and video-frame exports often append hashes, split prefixes, or
    frame counters. This key is intentionally conservative: it catches obvious
    duplicates without treating unrelated camera files as identical.
    """
    stem = image_path.stem.lower()
    stem = stem.split(".rf.")[0]
    stem = re.sub(r"^(?:src\d+_|pseudo\d+_)", "", stem)
    stem = re.sub(r"[_-](?:jpg|jpeg|png|bmp)$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def average_image_hash(image_path: Path, hash_size: int = 8) -> Optional[int]:
    """Small dependency-free perceptual hash used only for optional duplicate reports."""
    try:
        with Image.open(image_path).convert("L") as image:
            image = image.resize((hash_size, hash_size))
            pixels = list(image.getdata())
    except Exception:
        return None
    mean_value = sum(pixels) / max(len(pixels), 1)
    bits = 0
    for idx, pixel in enumerate(pixels):
        if pixel >= mean_value:
            bits |= 1 << idx
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def parse_group_list(value: str) -> Set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def scene_aware_yolo_split(
    source_root: Path,
    output_root: Path,
    val_groups_arg: str = "",
    test_groups_arg: str = "",
    logger: Optional[JsonRunLogger] = None,
) -> Path:
    """
    Rebuild an existing YOLO dataset with non-overlapping scene/video groups.

    This is important for SCI-style evaluation: the original Roboflow/random
    split can place neighboring video frames across train/valid/test and inflate
    mAP. This function pools all existing splits, groups images by source name,
    then assigns whole groups to train, valid, or test.
    """
    clear_dir(output_root)
    for split in ("train", "valid", "test"):
        ensure_dir(output_root / split / "images")
        ensure_dir(output_root / split / "labels")

    records: List[Tuple[Path, Path, str]] = []
    for split in ("train", "valid", "test"):
        for image_path in list_images(source_root / split / "images"):
            label_path = source_root / split / "labels" / f"{image_path.stem}.txt"
            valid, _, _ = count_label_file(label_path)
            if valid == 0:
                continue
            records.append((image_path, label_path, source_group_from_image_name(image_path.name)))

    groups = sorted({group for _, _, group in records})
    if len(groups) < 3:
        raise ValueError(f"Scene-aware split needs at least 3 source groups, found {groups}")

    group_counts = Counter(group for _, _, group in records)
    val_groups = parse_group_list(val_groups_arg)
    test_groups = parse_group_list(test_groups_arg)
    unknown = (val_groups | test_groups) - set(groups)
    if unknown:
        raise ValueError(f"Unknown scene groups {sorted(unknown)}. Available groups: {groups}")

    if not val_groups and not test_groups:
        ranked = [group for group, _ in group_counts.most_common()]
        test_count = max(1, round(len(ranked) * 0.25))
        val_count = max(1, round(len(ranked) * 0.15))
        test_groups = set(ranked[-test_count:])
        val_groups = set(ranked[-(test_count + val_count):-test_count])
    elif not val_groups:
        remaining = [group for group in groups if group not in test_groups]
        val_groups = {remaining[-1]}
    elif not test_groups:
        remaining = [group for group in groups if group not in val_groups]
        test_groups = {remaining[-1]}

    split_for_group = {}
    for group in groups:
        if group in test_groups:
            split_for_group[group] = "test"
        elif group in val_groups:
            split_for_group[group] = "valid"
        else:
            split_for_group[group] = "train"

    used_names = {split: set() for split in ("train", "valid", "test")}
    split_counts = Counter()
    for image_path, label_path, group in records:
        split = split_for_group[group]
        candidate_name = image_path.name
        stem = image_path.stem
        suffix = image_path.suffix
        counter = 1
        while candidate_name.lower() in used_names[split]:
            candidate_name = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names[split].add(candidate_name.lower())
        copy_file(image_path, output_root / split / "images" / candidate_name)
        copy_file(label_path, output_root / split / "labels" / f"{Path(candidate_name).stem}.txt")
        split_counts[split] += 1

    write_harmonization_metadata(output_root, f"{source_root.name}_scene_aware", list(CANONICAL_CLASSES.values()))
    data_yaml = write_data_yaml(output_root)
    payload = {
        "groups": groups,
        "group_counts": dict(group_counts),
        "split_for_group": split_for_group,
        "split_counts": dict(split_counts),
    }
    (output_root / "scene_split.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if logger:
        logger.log("scene_aware_split_created", root=output_root, **payload)
    print(f"Scene-aware split created: {output_root}")
    print(f"  groups: {split_for_group}")
    print(f"  counts: {dict(split_counts)}")
    return data_yaml


def find_coco_json(data_dir: Path) -> Optional[Path]:
    for pattern in ("*.json", "**/*.json"):
        for path in data_dir.glob(pattern):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if {"images", "annotations", "categories"}.issubset(data.keys()):
                return path
    return None


def convert_coco_to_yolo(coco_json: Path, output_root: Path) -> Path:
    """
    Convert COCO detection annotations to YOLO format.
    """
    clear_dir(output_root)
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    category_to_class = {}
    for category in data["categories"]:
        class_id = safe_class_id(category["name"])
        if class_id is not None:
            category_to_class[int(category["id"])] = class_id

    image_by_id = {int(item["id"]): item for item in data["images"]}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if int(ann["category_id"]) not in category_to_class:
            continue
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    source_root = coco_json.parent
    rows = list(image_by_id.items())
    random.Random(42).shuffle(rows)
    split_by_index = {}
    train_cut = int(len(rows) * 0.8)
    val_cut = int(len(rows) * 0.9)
    for idx, (image_id, _) in enumerate(rows):
        split_by_index[image_id] = "train" if idx < train_cut else "valid" if idx < val_cut else "test"

    for image_id, image_info in rows:
        anns = anns_by_image.get(image_id, [])
        if not anns:
            continue
        width = float(image_info["width"])
        height = float(image_info["height"])
        yolo_lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            class_id = category_to_class[int(ann["category_id"])]
            yolo_lines.append(
                f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}"
            )
        if not yolo_lines:
            continue
        split = split_by_index[image_id]
        image_path = source_root / image_info["file_name"]
        if not image_path.exists():
            matches = list(source_root.rglob(Path(image_info["file_name"]).name))
            if not matches:
                continue
            image_path = matches[0]
        dst_image = output_root / split / "images" / image_path.name
        dst_label = output_root / split / "labels" / f"{image_path.stem}.txt"
        copy_file(image_path, dst_image)
        ensure_dir(dst_label.parent)
        dst_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

    source_classes = [category["name"] for category in data["categories"]]
    write_harmonization_metadata(output_root, coco_json.parent.name, source_classes)
    return write_data_yaml(output_root)


def convert_classification_folders_to_yolo(data_dir: Path, output_root: Path) -> Path:
    """
    Convert class-folder image datasets to weak YOLO labels.

    Each image receives one full-image bounding box. This provides a runnable
    supervised baseline when only image-level labels are available.
    """
    clear_dir(output_root)
    class_images: List[Tuple[Path, int]] = []
    for image_path in list_images(data_dir):
        class_id = safe_class_id(image_path.parent.name)
        if class_id is None:
            continue
        class_images.append((image_path, class_id))

    if not class_images:
        raise ValueError(f"No class-folder images with recognized class names found in {data_dir}")

    random.Random(42).shuffle(class_images)
    train_cut = int(len(class_images) * 0.8)
    val_cut = int(len(class_images) * 0.9)

    for idx, (image_path, class_id) in enumerate(class_images):
        split = "train" if idx < train_cut else "valid" if idx < val_cut else "test"
        dst_image = output_root / split / "images" / image_path.name
        dst_label = output_root / split / "labels" / f"{image_path.stem}.txt"
        copy_file(image_path, dst_image)
        ensure_dir(dst_label.parent)
        dst_label.write_text(f"{class_id} 0.500000 0.500000 1.000000 1.000000\n", encoding="utf-8")

    print(
        "WARNING: Created weak full-image YOLO boxes from classification folders. "
        "Use real detection labels whenever possible."
    )
    source_classes = sorted({image_path.parent.name for image_path, _ in class_images})
    write_harmonization_metadata(output_root, data_dir.name, source_classes)
    return write_data_yaml(output_root)


def find_darknet_split_dirs(data_dir: Path) -> Dict[str, Path]:
    """
    Detect Darknet-style exports where each split folder contains images,
    same-stem .txt labels, and a _darknet.labels file.
    """
    split_dirs: Dict[str, Path] = {}
    candidates = list(data_dir.rglob("_darknet.labels"))
    for labels_file in candidates:
        folder = labels_file.parent
        name = folder.name.lower()
        parent_name = folder.parent.name.lower()
        if "train" in name or "train" in parent_name:
            split_dirs["train"] = folder
        elif "valid" in name or "val" in name or "valid" in parent_name or "val" in parent_name:
            split_dirs["valid"] = folder
        elif "test" in name or "test" in parent_name:
            split_dirs["test"] = folder
    return split_dirs


def convert_darknet_flat_to_yolo(data_dir: Path, output_root: Path) -> Path:
    """
    Convert Darknet flat split folders to canonical YOLO.

    Expected layout:
        train_rev2/train/*.jpg + *.txt + _darknet.labels
        valid_rev2/valid/*.jpg + *.txt + _darknet.labels
        test_rev2/test/*.jpg + *.txt + _darknet.labels
    """
    split_dirs = find_darknet_split_dirs(data_dir)
    if not split_dirs:
        raise ValueError(f"No Darknet split folders found in {data_dir}")

    clear_dir(output_root)
    source_classes: List[str] = []
    old_to_new_by_split: Dict[str, Dict[int, int]] = {}
    for split, folder in split_dirs.items():
        class_names = [
            line.strip()
            for line in (folder / "_darknet.labels").read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
        source_classes.extend(class_names)
        old_to_new = {}
        for old_id, class_name in enumerate(class_names):
            new_id = safe_class_id(class_name)
            if new_id is not None:
                old_to_new[old_id] = new_id
        old_to_new_by_split[split] = old_to_new

    copied = Counter()
    for split, folder in split_dirs.items():
        dst_images = output_root / split / "images"
        dst_labels = output_root / split / "labels"
        ensure_dir(dst_images)
        ensure_dir(dst_labels)
        old_to_new = old_to_new_by_split[split]
        for image_path in list_images(folder):
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                continue
            yolo_lines = []
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    old_id = int(float(parts[0]))
                    vals = [float(value) for value in parts[1:]]
                except ValueError:
                    continue
                if old_id not in old_to_new:
                    continue
                if not all(0.0 <= value <= 1.0 for value in vals) or vals[2] <= 0 or vals[3] <= 0:
                    continue
                yolo_lines.append(" ".join([str(old_to_new[old_id]), *[f"{value:.6f}" for value in vals]]))
            if not yolo_lines:
                continue
            copy_file(image_path, dst_images / image_path.name)
            (dst_labels / f"{image_path.stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            copied[split] += 1

    if not copied:
        raise ValueError(f"Darknet conversion produced no labeled images from {data_dir}")
    write_harmonization_metadata(output_root, data_dir.name, sorted(set(source_classes)))
    print(f"Detected Darknet flat dataset: {data_dir}; copied {dict(copied)}")
    return write_data_yaml(output_root)


def convert_to_yolo_format(data_dir: Path, output_root: Optional[Path] = None) -> Path:
    """
    Convert a dataset into canonical YOLO format.

    Supported inputs:
    - Existing YOLO dataset with data.yaml
    - COCO annotations JSON
    - Class-folder dataset, converted to weak full-image boxes
    """
    output_root = output_root or (data_dir.parent / f"{data_dir.name}_canonical_yolo")
    existing_yaml = detect_yolo_dataset(data_dir)
    if existing_yaml is not None:
        print(f"Detected YOLO dataset: {existing_yaml}")
        return remap_yolo_dataset(existing_yaml, output_root)

    coco_json = find_coco_json(data_dir)
    if coco_json is not None:
        print(f"Detected COCO annotations: {coco_json}")
        return convert_coco_to_yolo(coco_json, output_root)

    darknet_dirs = find_darknet_split_dirs(data_dir)
    if darknet_dirs:
        return convert_darknet_flat_to_yolo(data_dir, output_root)

    print(f"Trying class-folder conversion: {data_dir}")
    return convert_classification_folders_to_yolo(data_dir, output_root)


# =============================================================================
# E. YOLO dataset inspection and assembly
# =============================================================================

def write_data_yaml(dataset_root: Path) -> Path:
    ensure_dir(dataset_root)
    data_yaml = dataset_root / "data.yaml"
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(CANONICAL_CLASSES)}",
        "names:",
    ]
    for class_id in sorted(CANONICAL_CLASSES):
        lines.append(f"  {class_id}: {CANONICAL_CLASSES[class_id]}")
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return data_yaml


def write_harmonization_metadata(dataset_root: Path, source_name: str, source_class_names: Sequence[str]) -> Path:
    """
    Record source-to-canonical class harmonization and a class loss mask.

    Ultralytics YOLOv8 does not expose per-sample class loss masks through the
    plain Python train API, so this script keeps trainer compatibility and
    writes the mask as structured metadata. The mask documents which canonical
    classes are supervised by this source; missing classes can be ignored by a
    custom trainer or used for source-aware sampling later.
    """
    present_ids = sorted({safe_class_id(name) for name in source_class_names if safe_class_id(name) is not None})
    mask = {CANONICAL_CLASSES[class_id]: int(class_id in present_ids) for class_id in sorted(CANONICAL_CLASSES)}
    mapping = {}
    for name in source_class_names:
        class_id = safe_class_id(name)
        mapping[name] = CANONICAL_CLASSES[class_id] if class_id is not None else None
    metadata = {
        "source_name": source_name,
        "source_classes": list(source_class_names),
        "canonical_classes": CANONICAL_CLASSES,
        "source_to_canonical": mapping,
        "class_loss_mask": mask,
        "missing_canonical_classes": [name for name, enabled in mask.items() if not enabled],
    }
    path = dataset_root / "source_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def collect_dataset_classes(dataset_root: Path) -> List[str]:
    counts = {class_id: 0 for class_id in CANONICAL_CLASSES}
    for label_file in dataset_root.rglob("labels/*.txt"):
        _, _, class_counts = count_label_file(label_file)
        for class_id, count in class_counts.items():
            counts[class_id] += count
    return [CANONICAL_CLASSES[class_id] for class_id, count in counts.items() if count > 0]


def merge_source_metadata(source_roots: Sequence[Path], output_root: Path) -> Path:
    sources = []
    union_present: Set[str] = set()
    for root in source_roots:
        meta_path = root / "source_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            classes = collect_dataset_classes(root)
            meta = json.loads(write_harmonization_metadata(root, root.name, classes).read_text(encoding="utf-8"))
        observed_classes = set(collect_dataset_classes(root))
        meta["observed_class_loss_mask"] = {
            CANONICAL_CLASSES[class_id]: int(CANONICAL_CLASSES[class_id] in observed_classes)
            for class_id in sorted(CANONICAL_CLASSES)
        }
        sources.append(meta)
        for class_name, enabled in meta.get("class_loss_mask", {}).items():
            if enabled:
                union_present.add(class_name)
    combined = {
        "sources": sources,
        "canonical_classes": CANONICAL_CLASSES,
        "combined_class_loss_mask": {
            CANONICAL_CLASSES[class_id]: int(CANONICAL_CLASSES[class_id] in union_present)
            for class_id in sorted(CANONICAL_CLASSES)
        },
    }
    path = output_root / "source_metadata.json"
    path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def count_label_file(label_file: Path) -> Tuple[int, int, Dict[int, int]]:
    valid = 0
    bad = 0
    counts = {class_id: 0 for class_id in CANONICAL_CLASSES}
    if not label_file.exists():
        return 0, 0, counts
    for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            bad += 1
            continue
        try:
            class_id = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            bad += 1
            continue
        if class_id not in CANONICAL_CLASSES or not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            bad += 1
            continue
        valid += 1
        counts[class_id] += 1
    return valid, bad, counts


def inspect_yolo_dataset(dataset_root: Path) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    print(f"\nInspecting dataset: {dataset_root}")
    for split in ("train", "valid", "test"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        images = list_images(images_dir)
        labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        label_stems = {p.stem for p in labels}
        missing = sum(1 for img in images if img.stem not in label_stems)
        objects = 0
        bad_rows = 0
        empty = 0
        class_counts = {class_id: 0 for class_id in CANONICAL_CLASSES}
        for label in labels:
            valid, bad, counts = count_label_file(label)
            objects += valid
            bad_rows += bad
            empty += int(valid == 0)
            for class_id, count in counts.items():
                class_counts[class_id] += count
        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": objects,
            "empty_labels": empty,
            "missing_labels": missing,
            "bad_rows": bad_rows,
        }
        distribution = ", ".join(f"{CANONICAL_CLASSES[c]}={class_counts[c]}" for c in sorted(CANONICAL_CLASSES))
        print(
            f"  {split}: images={len(images)}, labels={len(labels)}, objects={objects}, "
            f"empty={empty}, missing={missing}, bad_rows={bad_rows}"
        )
        print(f"    by_class: {distribution}")
    return summary


def collect_yolo_image_records(dataset_root: Path, dataset_name: str) -> List[dict]:
    records: List[dict] = []
    for split in ("train", "valid", "test"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        for image_path in list_images(images_dir):
            label_path = labels_dir / f"{image_path.stem}.txt"
            valid, bad, class_counts = count_label_file(label_path)
            records.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "image": str(image_path),
                    "label": str(label_path) if label_path.exists() else "",
                    "normalized_base": normalized_base_filename(image_path),
                    "scene_group": source_group_from_image_name(image_path.name),
                    "objects": valid,
                    "bad_rows": bad,
                    **{f"boxes_{CANONICAL_CLASSES[class_id]}": count for class_id, count in class_counts.items()},
                }
            )
    return records


def write_rows_csv(path: Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_dataset_protocol_table(dataset_root: Path, output_csv: Path, protocol_role: str, main_conclusion: bool) -> None:
    rows = []
    for record in collect_yolo_image_records(dataset_root, dataset_root.name):
        rows.append(
            {
                "protocol_role": protocol_role,
                "main_conclusion": int(main_conclusion),
                "dataset": record["dataset"],
                "split": record["split"],
                "scene_group": record["scene_group"],
                "image": record["image"],
                "normalized_base": record["normalized_base"],
                "objects": record["objects"],
                **{key: value for key, value in record.items() if key.startswith("boxes_")},
            }
        )
    write_rows_csv(output_csv, rows)
    print(f"Saved protocol table: {output_csv}")


def write_duplicate_report(
    datasets: Sequence[Tuple[str, Path]],
    output_csv: Path,
    use_phash: bool = False,
    phash_threshold: int = 5,
) -> dict:
    records: List[dict] = []
    for dataset_name, dataset_root in datasets:
        records.extend(collect_yolo_image_records(dataset_root, dataset_name))

    by_base: Dict[str, List[dict]] = {}
    for record in records:
        by_base.setdefault(record["normalized_base"], []).append(record)

    rows = []
    duplicate_groups = 0
    for base, group in sorted(by_base.items()):
        datasets_in_group = sorted({item["dataset"] for item in group})
        if len(group) <= 1 or len(datasets_in_group) <= 1:
            continue
        duplicate_groups += 1
        for item in group:
            rows.append(
                {
                    "duplicate_type": "normalized_base_filename",
                    "duplicate_key": base,
                    "dataset": item["dataset"],
                    "split": item["split"],
                    "image": item["image"],
                    "scene_group": item["scene_group"],
                }
            )

    phash_groups = 0
    if use_phash:
        hash_records = []
        for record in records:
            image_hash = average_image_hash(Path(record["image"]))
            if image_hash is not None:
                hash_records.append((image_hash, record))
        for idx, (left_hash, left_record) in enumerate(hash_records):
            for right_hash, right_record in hash_records[idx + 1:]:
                if left_record["dataset"] == right_record["dataset"]:
                    continue
                distance = hamming_distance(left_hash, right_hash)
                if distance <= phash_threshold:
                    phash_groups += 1
                    rows.append(
                        {
                            "duplicate_type": "perceptual_hash",
                            "duplicate_key": f"hamming<={phash_threshold};distance={distance}",
                            "dataset": left_record["dataset"],
                            "split": left_record["split"],
                            "image": left_record["image"],
                            "scene_group": left_record["scene_group"],
                        }
                    )
                    rows.append(
                        {
                            "duplicate_type": "perceptual_hash",
                            "duplicate_key": f"hamming<={phash_threshold};distance={distance}",
                            "dataset": right_record["dataset"],
                            "split": right_record["split"],
                            "image": right_record["image"],
                            "scene_group": right_record["scene_group"],
                        }
                    )

    write_rows_csv(
        output_csv,
        rows,
        fieldnames=["duplicate_type", "duplicate_key", "dataset", "split", "image", "scene_group"],
    )
    summary = {
        "images_checked": len(records),
        "normalized_base_duplicate_groups": duplicate_groups,
        "phash_candidate_pairs": phash_groups if use_phash else None,
        "report": str(output_csv),
    }
    print(f"Saved duplicate report: {output_csv}")
    print(f"  duplicate groups by normalized filename: {duplicate_groups}")
    if use_phash:
        print(f"  perceptual-hash candidate pairs: {phash_groups}")
    return summary


def create_masked_external_dataset(source_root: Path, output_root: Path, allowed_class_ids: Sequence[int]) -> Path:
    """
    Build a strict external-test view for preprocessed-ffb.

    The preprocessed-ffb source does not annotate empty/overripe. For external
    evaluation, labels for missing classes must be absent and predictions for
    missing classes must be ignored by the custom masked evaluator.
    """
    clear_dir(output_root)
    allowed = set(allowed_class_ids)
    for split in ("train", "valid", "test"):
        ensure_dir(output_root / split / "images")
        ensure_dir(output_root / split / "labels")
        for image_path in list_images(source_root / split / "images"):
            label_path = source_root / split / "labels" / f"{image_path.stem}.txt"
            kept_lines = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    if class_id in allowed:
                        kept_lines.append(" ".join([str(class_id), *parts[1:]]))
            if not kept_lines:
                continue
            copy_file(image_path, output_root / split / "images" / image_path.name)
            (output_root / split / "labels" / f"{image_path.stem}.txt").write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    write_harmonization_metadata(output_root, "preprocessed_ffb_masked_external", PREPROCESSED_FFB_EVAL_CLASSES)
    return write_data_yaml(output_root)


def create_class_balanced_yolo_dataset(
    source_root: Path,
    output_root: Path,
    split: str = "train",
    target_images_per_class: int = 0,
    allowed_class_ids: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> Path:
    """
    Create an oversampled YOLO dataset view for class-balanced adaptation.

    Images are copied into the requested split, duplicating minority-class
    images with deterministic suffixes until every class has at least the
    target image count. Labels are unchanged; this keeps Ultralytics trainer
    compatibility while reducing dominance of easy/common external classes.
    """
    clear_dir(output_root)
    for dst_split in ("train", "valid", "test"):
        ensure_dir(output_root / dst_split / "images")
        ensure_dir(output_root / dst_split / "labels")

    allowed = set(allowed_class_ids) if allowed_class_ids is not None else set(CANONICAL_CLASSES)
    rng = random.Random(seed)
    class_to_images: Dict[int, List[Tuple[Path, Path]]] = {class_id: [] for class_id in allowed}
    source_images = list_images(source_root / split / "images")
    for image_path in source_images:
        label_path = source_root / split / "labels" / f"{image_path.stem}.txt"
        valid, _, counts = count_label_file(label_path)
        if valid == 0:
            continue
        present = [class_id for class_id, count in counts.items() if count > 0 and class_id in allowed]
        for class_id in present:
            class_to_images[class_id].append((image_path, label_path))

    if target_images_per_class <= 0:
        target_images_per_class = max((len(items) for items in class_to_images.values()), default=0)

    rows = []
    used_names: Set[str] = set()
    dst_images = output_root / split / "images"
    dst_labels = output_root / split / "labels"
    for class_id, items in sorted(class_to_images.items()):
        if not items:
            rows.append({"class_id": class_id, "class_name": CANONICAL_CLASSES[class_id], "source_images": 0, "sampled_images": 0})
            continue
        sampled = list(items)
        while len(sampled) < target_images_per_class:
            sampled.append(rng.choice(items))
        sampled = sampled[:target_images_per_class]
        rows.append(
            {
                "class_id": class_id,
                "class_name": CANONICAL_CLASSES[class_id],
                "source_images": len(items),
                "sampled_images": len(sampled),
            }
        )
        for sample_idx, (image_path, label_path) in enumerate(sampled):
            candidate_name = f"bal_c{class_id}_{sample_idx:05d}_{image_path.name}"
            while candidate_name.lower() in used_names:
                candidate_name = f"bal_c{class_id}_{sample_idx:05d}_{rng.randrange(10**9)}_{image_path.name}"
            used_names.add(candidate_name.lower())
            link_or_copy_file(image_path, dst_images / candidate_name)
            link_or_copy_file(label_path, dst_labels / f"{Path(candidate_name).stem}.txt")

    for passthrough_split in ("valid", "test"):
        copy_yolo_split(source_root, passthrough_split, output_root)
    write_rows_csv(output_root / "class_balance_report.csv", rows)
    write_harmonization_metadata(output_root, f"{source_root.name}_class_balanced", collect_dataset_classes(source_root))
    return write_data_yaml(output_root)


def write_pseudo_ablation_table(output_csv: Path) -> None:
    rows = [
        {
            "ablation": "full_ssod",
            "external_domain_adaptation": 0,
            "multi_view_consistency": 1,
            "quality_scoring": 1,
            "dynamic_thresholds": 1,
            "balanced_per_class_selection": 1,
            "purpose": "main proposed pseudo-label pipeline",
        },
        {
            "ablation": "model2_external_domain_adaptation",
            "external_domain_adaptation": 1,
            "multi_view_consistency": 0,
            "quality_scoring": 0,
            "dynamic_thresholds": 0,
            "balanced_per_class_selection": 0,
            "purpose": "measure supervised external-domain adaptation effect",
        },
        {
            "ablation": "model2_plus_full_ssod",
            "external_domain_adaptation": 1,
            "multi_view_consistency": 1,
            "quality_scoring": 1,
            "dynamic_thresholds": 1,
            "balanced_per_class_selection": 1,
            "purpose": "measure whether quality-controlled SSOD adds value after external-domain adaptation",
        },
        {
            "ablation": "no_external_domain_adaptation",
            "external_domain_adaptation": 0,
            "multi_view_consistency": 0,
            "quality_scoring": 0,
            "dynamic_thresholds": 0,
            "balanced_per_class_selection": 0,
            "purpose": "baseline trained only on ScienceDB/OILPALM",
        },
        {
            "ablation": "no_consistency",
            "external_domain_adaptation": 0,
            "multi_view_consistency": 0,
            "quality_scoring": 1,
            "dynamic_thresholds": 1,
            "balanced_per_class_selection": 1,
            "purpose": "measure consistency-training contribution",
        },
        {
            "ablation": "no_quality_score",
            "external_domain_adaptation": 0,
            "multi_view_consistency": 1,
            "quality_scoring": 0,
            "dynamic_thresholds": 1,
            "balanced_per_class_selection": 1,
            "purpose": "measure pseudo-label quality score contribution",
        },
        {
            "ablation": "fixed_threshold",
            "external_domain_adaptation": 0,
            "multi_view_consistency": 1,
            "quality_scoring": 1,
            "dynamic_thresholds": 0,
            "balanced_per_class_selection": 1,
            "purpose": "measure adaptive threshold contribution",
        },
        {
            "ablation": "unbalanced_selection",
            "external_domain_adaptation": 0,
            "multi_view_consistency": 1,
            "quality_scoring": 1,
            "dynamic_thresholds": 1,
            "balanced_per_class_selection": 0,
            "purpose": "measure class-balance contribution",
        },
    ]
    write_rows_csv(output_csv, rows)
    print(f"Saved pseudo-label ablation design table: {output_csv}")


def copy_yolo_split(source_root: Path, split: str, dest_root: Path, prefix: str = "") -> int:
    src_images = source_root / split / "images"
    src_labels = source_root / split / "labels"
    dst_images = dest_root / split / "images"
    dst_labels = dest_root / split / "labels"
    ensure_dir(dst_images)
    ensure_dir(dst_labels)
    copied = 0
    used_names = {p.name.lower() for p in dst_images.glob("*") if p.is_file()}
    for image_path in list_images(src_images):
        label_path = src_labels / f"{image_path.stem}.txt"
        valid, _, _ = count_label_file(label_path)
        if valid == 0:
            continue
        candidate_name = f"{prefix}{image_path.name}"
        stem = Path(candidate_name).stem
        suffix = image_path.suffix
        counter = 1
        while candidate_name.lower() in used_names or (dst_images / candidate_name).exists():
            candidate_name = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate_name.lower())
        dst_image = dst_images / candidate_name
        dst_label = dst_labels / f"{Path(candidate_name).stem}.txt"
        link_or_copy_file(image_path, dst_image)
        link_or_copy_file(label_path, dst_label)
        copied += 1
    return copied


def assemble_combined_dataset(old_data_dir: Path, pseudo_labels: Sequence[Tuple[Path, Path]], output_root: Path) -> Path:
    """
    Build a YOLO dataset from supervised data plus pseudo-labeled images.
    """
    clear_dir(output_root)
    for split in ("train", "valid", "test"):
        ensure_dir(output_root / split / "images")
        ensure_dir(output_root / split / "labels")

    original_counts = {split: copy_yolo_split(old_data_dir, split, output_root) for split in ("train", "valid", "test")}
    pseudo_counts = {"train": 0, "valid": 0, "test": 0}

    for source_idx, (pseudo_images, pseudo_label_dir) in enumerate(pseudo_labels, start=1):
        images = list_images(pseudo_images)
        random.Random(42 + source_idx).shuffle(images)
        train_cut = int(len(images) * 0.80)
        val_cut = int(len(images) * 0.90)
        split_map = {
            "train": images[:train_cut],
            "valid": images[train_cut:val_cut],
            "test": images[val_cut:],
        }
        for split, split_images in split_map.items():
            dst_images = output_root / split / "images"
            dst_labels = output_root / split / "labels"
            used_names = {p.name.lower() for p in dst_images.glob("*") if p.is_file()}
            for image_path in split_images:
                label_path = pseudo_label_dir / f"{image_path.stem}.txt"
                valid, _, _ = count_label_file(label_path)
                if valid == 0:
                    continue
                candidate_name = f"pseudo{source_idx}_{image_path.name}"
                stem = Path(candidate_name).stem
                suffix = image_path.suffix
                counter = 1
                while candidate_name.lower() in used_names:
                    candidate_name = f"{stem}_{counter}{suffix}"
                    counter += 1
                used_names.add(candidate_name.lower())
                copy_file(image_path, dst_images / candidate_name)
                copy_file(label_path, dst_labels / f"{Path(candidate_name).stem}.txt")
                pseudo_counts[split] += 1

    data_yaml = write_data_yaml(output_root)
    merge_source_metadata([old_data_dir], output_root)
    print("\nCombined dataset assembled:")
    for split in ("train", "valid", "test"):
        print(f"  {split}: original={original_counts[split]}, pseudo={pseudo_counts[split]}")
    inspect_yolo_dataset(output_root)
    return data_yaml


# =============================================================================
# F. Augmentations, consistency matching, and quality scoring
# =============================================================================

def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def xywh_to_xyxy(box: BoxPrediction) -> Tuple[float, float, float, float]:
    return (
        box.x_center - box.width / 2.0,
        box.y_center - box.height / 2.0,
        box.x_center + box.width / 2.0,
        box.y_center + box.height / 2.0,
    )


def box_iou(a: BoxPrediction, b: BoxPrediction) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = a.width * a.height + b.width * b.height - intersection
    return 0.0 if union <= 0.0 else intersection / union


def generate_augmented_views(image: Image.Image, augmentation_params: Sequence[str]) -> List[Tuple[str, Image.Image, dict]]:
    """
    Create strong views for SSOD consistency training.

    Each returned transform dict is used to map predicted boxes back to the
    original image coordinates.
    """
    views: List[Tuple[str, Image.Image, dict]] = []
    width, height = image.size
    for aug_name in augmentation_params:
        if aug_name == "brightness":
            views.append((aug_name, ImageEnhance.Brightness(image).enhance(1.30), {"type": "identity"}))
        elif aug_name == "contrast":
            views.append((aug_name, ImageEnhance.Contrast(image).enhance(1.35), {"type": "identity"}))
        elif aug_name == "hflip":
            views.append((aug_name, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), {"type": "hflip"}))
        elif aug_name == "vflip":
            views.append((aug_name, image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), {"type": "vflip"}))
        elif aug_name == "random_crop":
            crop_w = int(width * 0.85)
            crop_h = int(height * 0.85)
            x0 = max(0, (width - crop_w) // 2)
            y0 = max(0, (height - crop_h) // 2)
            crop = image.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((width, height))
            views.append((aug_name, crop, {"type": "crop", "x0": x0, "y0": y0, "cw": crop_w, "ch": crop_h, "w": width, "h": height}))
        elif aug_name == "masked_patches":
            # MIC-style context consistency: hide deterministic patches while
            # preserving the image coordinate system for direct IoU matching.
            masked = image.copy()
            fill = tuple(int(value) for value in image.resize((1, 1)).getpixel((0, 0)))
            patch_w = max(1, int(width * 0.18))
            patch_h = max(1, int(height * 0.18))
            for cx, cy in ((0.30, 0.34), (0.68, 0.62), (0.52, 0.48)):
                x0 = max(0, min(width - patch_w, int(width * cx - patch_w / 2)))
                y0 = max(0, min(height - patch_h, int(height * cy - patch_h / 2)))
                masked.paste(fill, (x0, y0, x0 + patch_w, y0 + patch_h))
            views.append((aug_name, masked, {"type": "identity"}))
    return views


def invert_box_transform(box: BoxPrediction, transform: dict) -> BoxPrediction:
    transform_type = transform.get("type")
    if transform_type == "identity":
        return box
    if transform_type == "hflip":
        return BoxPrediction(box.class_id, 1.0 - box.x_center, box.y_center, box.width, box.height, box.confidence)
    if transform_type == "vflip":
        return BoxPrediction(box.class_id, box.x_center, 1.0 - box.y_center, box.width, box.height, box.confidence)
    if transform_type == "crop":
        x0, y0, cw, ch, w, h = transform["x0"], transform["y0"], transform["cw"], transform["ch"], transform["w"], transform["h"]
        return BoxPrediction(
            class_id=box.class_id,
            x_center=clamp01((x0 + box.x_center * cw) / w),
            y_center=clamp01((y0 + box.y_center * ch) / h),
            width=clamp01(box.width * cw / w),
            height=clamp01(box.height * ch / h),
            confidence=box.confidence,
        )
    return box


def result_to_boxes(result) -> List[BoxPrediction]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    predictions = []
    for xywh, cls_value, conf_value in zip(boxes.xywhn.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
        predictions.append(
            BoxPrediction(
                class_id=int(cls_value),
                x_center=clamp01(xywh[0]),
                y_center=clamp01(xywh[1]),
                width=clamp01(xywh[2]),
                height=clamp01(xywh[3]),
                confidence=float(conf_value),
            )
        )
    return predictions


def predict_one(model: YOLO, image_path: Path, imgsz: int, device: str, conf: float) -> List[BoxPrediction]:
    results = model.predict(source=str(image_path), imgsz=imgsz, device=device, conf=conf, save=False, verbose=False)
    return result_to_boxes(results[0]) if results else []


def consistency_support(box: BoxPrediction, augmented_boxes: Sequence[List[BoxPrediction]], iou_threshold: float) -> int:
    support = 1
    for view_boxes in augmented_boxes:
        if any(candidate.class_id == box.class_id and box_iou(box, candidate) >= iou_threshold for candidate in view_boxes):
            support += 1
    return support


def ensemble_support(
    box: BoxPrediction,
    ensemble_boxes: Sequence[List[BoxPrediction]],
    iou_threshold: float,
    allowed_class_ids: Optional[Set[int]] = None,
) -> int:
    """Count teacher votes whose boxes overlap and whose classes are allowed."""
    support = 1
    allowed = allowed_class_ids or {box.class_id}
    for teacher_boxes in ensemble_boxes:
        if any(candidate.class_id in allowed and box_iou(box, candidate) >= iou_threshold for candidate in teacher_boxes):
            support += 1
    return support


def compute_quality_score(
    prediction: BoxPrediction,
    image_size_value: Tuple[int, int],
    expected_class_id: Optional[int],
    class_thresholds: Dict[int, float],
    min_area: float,
    max_area: float,
    max_aspect_ratio: float,
    edge_margin: float,
) -> BoxPrediction:
    """
    Composite Q score = f(confidence, geometry, edge, class consistency).

    This follows the SSOD intuition that high confidence alone is not enough:
    pseudo boxes should also have plausible geometry, avoid ambiguous borders,
    and agree with weak metadata when available.
    """
    reasons: List[str] = []
    threshold = class_thresholds.get(prediction.class_id, 0.5)
    confidence_score = min(1.0, prediction.confidence / max(threshold, 1e-6))
    if prediction.confidence < threshold:
        reasons.append("below_dynamic_threshold")

    area = prediction.width * prediction.height
    if area < min_area:
        geometry_area_score = max(0.0, area / min_area)
        reasons.append("small_box")
    elif area > max_area:
        geometry_area_score = max(0.0, 1.0 - (area - max_area) / max(max_area, 1e-6))
        reasons.append("large_box")
    else:
        geometry_area_score = 1.0

    if prediction.width <= 0.0 or prediction.height <= 0.0:
        aspect_score = 0.0
        reasons.append("non_positive_box")
    else:
        aspect = max(prediction.width / prediction.height, prediction.height / prediction.width)
        aspect_score = max(0.0, 1.0 - (aspect - 1.0) / max(max_aspect_ratio - 1.0, 1e-6))
        if aspect > max_aspect_ratio:
            reasons.append("bad_aspect_ratio")

    x1, y1, x2, y2 = xywh_to_xyxy(prediction)
    touches_edge = x1 < edge_margin or y1 < edge_margin or x2 > 1.0 - edge_margin or y2 > 1.0 - edge_margin
    edge_score = 0.0 if touches_edge else 1.0
    if touches_edge:
        reasons.append("edge_contact")

    if expected_class_id is None:
        class_consistency_score = 0.75
    elif expected_class_id == prediction.class_id:
        class_consistency_score = 1.0
    else:
        class_consistency_score = 0.0
        reasons.append("folder_class_mismatch")

    geometry_score = 0.70 * geometry_area_score + 0.30 * aspect_score
    prediction.quality_score = (
        0.42 * confidence_score
        + 0.25 * geometry_score
        + 0.15 * edge_score
        + 0.18 * class_consistency_score
    )
    prediction.reasons = reasons
    return prediction


# =============================================================================
# G. Dynamic class thresholds and metrics
# =============================================================================

def dynamic_class_threshold_update(metrics, base_threshold: float, min_threshold: float, max_threshold: float, target_map: float) -> Dict[int, float]:
    """
    Make difficult classes stricter and easy classes looser.

    The common practical SSOD problem is confirmation bias: weak classes accept
    noisy pseudo labels too easily. We raise their thresholds based on lower
    validation AP.
    """
    thresholds = {class_id: base_threshold for class_id in CANONICAL_CLASSES}
    maps = getattr(metrics.box, "maps", None)
    if maps is None:
        return thresholds
    for class_id in CANONICAL_CLASSES:
        class_ap = float(maps[class_id]) if class_id < len(maps) else target_map
        thresholds[class_id] = max(min_threshold, min(max_threshold, base_threshold + max(0.0, target_map - class_ap) * 0.30))
    return thresholds


def per_class_metric_summary(metrics) -> Dict[str, dict]:
    """Extract class-level AP where Ultralytics exposes it."""
    maps = getattr(metrics.box, "maps", None)
    summary: Dict[str, dict] = {}
    for class_id, class_name in CANONICAL_CLASSES.items():
        ap5095 = float(maps[class_id]) if maps is not None and class_id < len(maps) else None
        summary[class_name] = {"ap50_95": ap5095}
    return summary


def metrics_to_stage(stage: str, metrics, model_path: Path, data_yaml: Path) -> StageMetrics:
    return StageMetrics(
        stage=stage,
        precision=float(metrics.box.mp),
        recall=float(metrics.box.mr),
        map50=float(metrics.box.map50),
        map5095=float(metrics.box.map),
        model_path=str(model_path),
        data_yaml=str(data_yaml),
    )


def print_thresholds(thresholds: Dict[int, float]) -> None:
    text = ", ".join(f"{CANONICAL_CLASSES[c]}={thresholds[c]:.3f}" for c in sorted(thresholds))
    print(f"Dynamic thresholds: {text}")


def apply_class_threshold_overrides(thresholds: Dict[int, float], overrides: Dict[int, float]) -> Dict[int, float]:
    """Apply explicit hard-class pseudo-label thresholds after dynamic update."""
    if not overrides:
        return thresholds
    updated = dict(thresholds)
    updated.update(overrides)
    readable = ", ".join(f"{CANONICAL_CLASSES[class_id]}={value:.3f}" for class_id, value in sorted(overrides.items()))
    print(f"Class threshold overrides: {readable}")
    return updated


# =============================================================================
# H. Pseudo-label generation
# =============================================================================

def write_pseudo_label(label_path: Path, boxes: Sequence[BoxPrediction], include_debug: bool = False) -> None:
    ensure_dir(label_path.parent)
    lines = []
    for box in boxes:
        line = f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        if include_debug:
            line += f" conf={box.confidence:.4f} q={box.quality_score:.4f} support={box.support} ensemble_support={box.ensemble_support} reasons={'|'.join(box.reasons)}"
        lines.append(line)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def generate_pseudo_labels_for_scene(
    model_path: Path,
    image_paths: Sequence[Path],
    image_root: Path,
    output_root: Path,
    class_thresholds: Dict[int, float],
    args: argparse.Namespace,
    iteration: int,
    scene: str,
) -> Tuple[Path, Path, List[ImagePseudoResult]]:
    """
    Generate consistency-filtered and quality-scored pseudo labels for one scene.
    """
    scene_root = output_root / f"iter_{iteration}_{scene}"
    accepted_images_dir = scene_root / "accepted_images"
    accepted_labels_dir = scene_root / "accepted_labels"
    rejected_labels_dir = scene_root / "rejected_debug_labels"
    clear_dir(accepted_images_dir)
    clear_dir(accepted_labels_dir)
    clear_dir(rejected_labels_dir)

    model = YOLO(str(model_path))
    ensemble_models = [YOLO(str(path)) for path in args.ensemble_teacher_models]
    min_conf = min(class_thresholds.values())
    if args.consistency_profile == "mic_masked":
        aug_names = ["masked_patches", "brightness", "contrast", "hflip", "random_crop"]
    else:
        aug_names = ["brightness", "contrast", "hflip", "random_crop"]
    if args.use_vertical_flip:
        aug_names.append("vflip")

    results: List[ImagePseudoResult] = []
    with tempfile.TemporaryDirectory(prefix="ffb_ssod_aug_") as temp_name:
        temp_dir = Path(temp_name)
        for idx, image_path in enumerate(image_paths, start=1):
            original_boxes = predict_one(model, image_path, imgsz=args.imgsz, device=args.device, conf=min_conf)
            ensemble_boxes = [
                predict_one(ensemble_model, image_path, imgsz=args.imgsz, device=args.device, conf=min_conf)
                for ensemble_model in ensemble_models
            ]
            expected_class_id = expected_class_from_folder(image_path, image_root)
            augmented_boxes: List[List[BoxPrediction]] = []
            with Image.open(image_path).convert("RGB") as image:
                views = generate_augmented_views(image, aug_names[: args.num_aug_views])
                for aug_name, aug_image, transform in views:
                    aug_path = temp_dir / f"{image_path.stem}_{aug_name}.jpg"
                    aug_image.save(aug_path, quality=95)
                    view_boxes = predict_one(model, aug_path, imgsz=args.imgsz, device=args.device, conf=min_conf)
                    augmented_boxes.append([invert_box_transform(box, transform) for box in view_boxes])

            accepted: List[BoxPrediction] = []
            rejected: List[BoxPrediction] = []
            for box in original_boxes:
                box.support = consistency_support(box, augmented_boxes, args.consistency_iou)
                box.ensemble_support = ensemble_support(box, ensemble_boxes, args.ensemble_iou)
                scored = compute_quality_score(
                    prediction=box,
                    image_size_value=image_size(image_path),
                    expected_class_id=expected_class_id,
                    class_thresholds=class_thresholds,
                    min_area=args.min_box_area,
                    max_area=args.max_box_area,
                    max_aspect_ratio=args.max_aspect_ratio,
                    edge_margin=args.edge_margin,
                )
                consistency_ok = args.disable_consistency or scored.support >= args.min_consistency_support
                class_quality_threshold = args.quality_threshold_by_class.get(scored.class_id, args.quality_threshold)
                quality_ok = args.disable_quality_scoring or scored.quality_score >= class_quality_threshold
                folder_ok = args.allow_folder_mismatch or "folder_class_mismatch" not in scored.reasons
                edge_ok = not args.reject_edge_contact or "edge_contact" not in scored.reasons
                ensemble_ok = not ensemble_models or scored.ensemble_support >= args.min_ensemble_support
                hard_relabel_ok = False
                ordinal_rescue_ok = False
                if (
                    args.enable_ordinal_under_ripe_rescue
                    and expected_class_id == CLASS_NAME_TO_ID["under_ripe"]
                    and scored.class_id in {CLASS_NAME_TO_ID["ripe"], CLASS_NAME_TO_ID["unripe"]}
                    and scored.confidence >= args.ordinal_min_conf
                    and scored.quality_score >= args.ordinal_min_quality
                    and scored.support >= args.ordinal_min_view_support
                ):
                    ordinal_vote_classes = {
                        CLASS_NAME_TO_ID["ripe"],
                        CLASS_NAME_TO_ID["under_ripe"],
                        CLASS_NAME_TO_ID["unripe"],
                    }
                    ordinal_votes = ensemble_support(scored, ensemble_boxes, args.ensemble_iou, ordinal_vote_classes)
                    if ordinal_votes >= args.ordinal_min_ensemble_support:
                        original_class_id = scored.class_id
                        scored.class_id = CLASS_NAME_TO_ID["under_ripe"]
                        scored.ensemble_support = ordinal_votes
                        scored.reasons = [
                            reason for reason in scored.reasons
                            if reason not in {"folder_class_mismatch", "below_dynamic_threshold"}
                        ]
                        scored.reasons.append(f"ordinal_under_ripe_rescue_from_{CANONICAL_CLASSES[original_class_id]}")
                        ordinal_rescue_ok = True
                if (
                    args.enable_folder_guided_hard_class_relabel
                    and expected_class_id in args.hard_class_ids
                    and scored.class_id in args.hard_class_source_map.get(expected_class_id, set())
                    and scored.confidence >= args.hard_class_min_conf
                    and scored.quality_score >= args.hard_class_min_quality
                    and scored.support >= args.hard_class_min_support
                    and "edge_contact" not in scored.reasons
                ):
                    original_class_id = scored.class_id
                    scored.class_id = expected_class_id
                    scored.reasons = [
                        reason for reason in scored.reasons
                        if reason not in {"folder_class_mismatch", "below_dynamic_threshold"}
                    ]
                    scored.reasons.append(f"folder_guided_relabel_from_{CANONICAL_CLASSES[original_class_id]}")
                    hard_relabel_ok = True

                if (consistency_ok and quality_ok and folder_ok and ensemble_ok and edge_ok) or hard_relabel_ok or ordinal_rescue_ok:
                    accepted.append(scored)
                else:
                    if not consistency_ok:
                        scored.reasons.append("low_consistency_support")
                    if not quality_ok:
                        scored.reasons.append(f"low_quality_score<{class_quality_threshold:.3f}")
                    if not ensemble_ok:
                        scored.reasons.append(f"low_ensemble_support<{args.min_ensemble_support}")
                    if not edge_ok:
                        scored.reasons.append("strict_edge_rejection")
                    rejected.append(scored)

            if len(accepted) > args.max_boxes_per_image:
                accepted.sort(key=lambda box: (box.quality_score, box.confidence, box.support), reverse=True)
                overflow = accepted[args.max_boxes_per_image :]
                accepted = accepted[: args.max_boxes_per_image]
                for box in overflow:
                    box.reasons.append("per_image_top_quality_truncation")
                rejected.extend(overflow)

            if accepted:
                dst_image = accepted_images_dir / image_path.name
                dst_label = accepted_labels_dir / f"{image_path.stem}.txt"
                copy_file(image_path, dst_image)
                write_pseudo_label(dst_label, accepted, include_debug=False)
                status = "accepted"
                reasons: List[str] = []
            else:
                write_pseudo_label(rejected_labels_dir / f"{image_path.stem}.txt", rejected, include_debug=True)
                status = "rejected"
                reasons = sorted({reason for box in rejected for reason in box.reasons}) or ["no_valid_box"]

            results.append(ImagePseudoResult(image_path, scene, accepted, rejected, status, reasons))
            if idx % 50 == 0:
                print(f"  scene={scene} processed {idx}/{len(image_paths)}")

    apply_balanced_pseudo_label_selection(
        results,
        accepted_images_dir=accepted_images_dir,
        accepted_labels_dir=accepted_labels_dir,
        max_images_per_class=args.max_pseudo_images_per_class,
        max_boxes_per_class=args.max_pseudo_boxes_per_class,
        max_boxes_per_class_overrides=args.max_pseudo_boxes_per_class_overrides,
    )
    report_pseudo_counts(results, iteration=iteration, scene=scene)
    write_pseudo_report(scene_root / "pseudo_report.csv", results)
    return accepted_images_dir, accepted_labels_dir, results


def write_pseudo_report(report_path: Path, results: Sequence[ImagePseudoResult]) -> None:
    ensure_dir(report_path.parent)
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "scene", "status", "class", "conf", "quality", "support", "ensemble_support", "reasons"],
        )
        writer.writeheader()
        for result in results:
            boxes = result.accepted_boxes or result.rejected_boxes
            if not boxes:
                writer.writerow({"image": str(result.image_path), "scene": result.scene, "status": result.status, "class": "", "conf": "", "quality": "", "support": "", "ensemble_support": "", "reasons": "|".join(result.reasons)})
            for box in boxes:
                writer.writerow(
                    {
                        "image": str(result.image_path),
                        "scene": result.scene,
                        "status": result.status,
                        "class": CANONICAL_CLASSES.get(box.class_id, "unknown"),
                        "conf": f"{box.confidence:.4f}",
                        "quality": f"{box.quality_score:.4f}",
                        "support": box.support,
                        "ensemble_support": box.ensemble_support,
                        "reasons": "|".join(sorted(set(box.reasons))),
                    }
                )


def report_pseudo_counts(results: Sequence[ImagePseudoResult], iteration: int, scene: str) -> None:
    accepted_images = sum(1 for result in results if result.status == "accepted")
    rejected_images = len(results) - accepted_images
    accepted_counts = {class_id: 0 for class_id in CANONICAL_CLASSES}
    rejected_counts = {class_id: 0 for class_id in CANONICAL_CLASSES}
    for result in results:
        for box in result.accepted_boxes:
            accepted_counts[box.class_id] += 1
        for box in result.rejected_boxes:
            if box.class_id in rejected_counts:
                rejected_counts[box.class_id] += 1
    print(f"\nIteration {iteration} scene {scene} pseudo-label summary:")
    print(f"  accepted_images={accepted_images}, rejected_images={rejected_images}")
    print("  accepted_boxes: " + ", ".join(f"{CANONICAL_CLASSES[c]}={accepted_counts[c]}" for c in sorted(accepted_counts)))
    print("  rejected_boxes: " + ", ".join(f"{CANONICAL_CLASSES[c]}={rejected_counts[c]}" for c in sorted(rejected_counts)))


def summarize_pseudo_results(results: Sequence[ImagePseudoResult]) -> dict:
    accepted_counts = {CANONICAL_CLASSES[class_id]: 0 for class_id in CANONICAL_CLASSES}
    rejected_counts = {CANONICAL_CLASSES[class_id]: 0 for class_id in CANONICAL_CLASSES}
    reasons: Dict[str, int] = {}
    for result in results:
        for box in result.accepted_boxes:
            accepted_counts[CANONICAL_CLASSES.get(box.class_id, "unknown")] = accepted_counts.get(CANONICAL_CLASSES.get(box.class_id, "unknown"), 0) + 1
        for box in result.rejected_boxes:
            class_name = CANONICAL_CLASSES.get(box.class_id, "unknown")
            rejected_counts[class_name] = rejected_counts.get(class_name, 0) + 1
            for reason in box.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        for reason in result.reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "accepted_images": sum(1 for result in results if result.status == "accepted"),
        "rejected_images": sum(1 for result in results if result.status != "accepted"),
        "accepted_boxes_by_class": accepted_counts,
        "rejected_boxes_by_class": rejected_counts,
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda item: item[1], reverse=True)),
    }


def apply_balanced_pseudo_label_selection(
    results: List[ImagePseudoResult],
    accepted_images_dir: Path,
    accepted_labels_dir: Path,
    max_images_per_class: int,
    max_boxes_per_class: int,
    max_boxes_per_class_overrides: Dict[int, int],
) -> None:
    """
    Cap accepted pseudo-label images per class for each iteration/scene.

    This reduces confirmation bias when one easy class dominates pseudo labels.
    Images containing multiple classes are admitted only if every class in that
    image still has remaining quota.
    """
    if max_images_per_class <= 0 and max_boxes_per_class <= 0 and not max_boxes_per_class_overrides:
        return
    accepted_results = [result for result in results if result.status == "accepted" and result.accepted_boxes]
    accepted_results.sort(
        key=lambda result: sum(box.quality_score for box in result.accepted_boxes) / max(len(result.accepted_boxes), 1),
        reverse=True,
    )
    used_by_class = {class_id: 0 for class_id in CANONICAL_CLASSES}
    used_boxes_by_class = {class_id: 0 for class_id in CANONICAL_CLASSES}
    selected: Set[Path] = set()
    for result in accepted_results:
        image_classes = {box.class_id for box in result.accepted_boxes}
        box_counts = Counter(box.class_id for box in result.accepted_boxes)
        image_quota_ok = max_images_per_class <= 0 or all(
            used_by_class.get(class_id, 0) < max_images_per_class for class_id in image_classes
        )
        box_quota_ok = all(
            max_boxes_per_class_overrides.get(class_id, max_boxes_per_class) <= 0
            or used_boxes_by_class.get(class_id, 0) + count
            <= max_boxes_per_class_overrides.get(class_id, max_boxes_per_class)
            for class_id, count in box_counts.items()
        )
        if image_quota_ok and box_quota_ok:
            selected.add(result.image_path)
            for class_id in image_classes:
                used_by_class[class_id] += 1
            for class_id, count in box_counts.items():
                used_boxes_by_class[class_id] += count

    for result in accepted_results:
        if result.image_path in selected:
            continue
        result.rejected_boxes.extend(result.accepted_boxes)
        for box in result.rejected_boxes:
            box.reasons.append("balanced_class_quota")
        result.accepted_boxes = []
        result.status = "rejected"
        result.reasons = sorted(set([*result.reasons, "balanced_class_quota"]))
        image_file = accepted_images_dir / result.image_path.name
        label_file = accepted_labels_dir / f"{result.image_path.stem}.txt"
        if image_file.exists():
            image_file.unlink()
        if label_file.exists():
            label_file.unlink()
    print(
        "Balanced pseudo-label image quotas: "
        + ", ".join(f"{CANONICAL_CLASSES[class_id]}={used_by_class[class_id]}/{max_images_per_class}" for class_id in sorted(used_by_class))
    )
    print(
        "Balanced pseudo-label box quotas: "
        + ", ".join(
            f"{CANONICAL_CLASSES[class_id]}={used_boxes_by_class[class_id]}/"
            f"{max_boxes_per_class_overrides.get(class_id, max_boxes_per_class)}"
            for class_id in sorted(used_boxes_by_class)
        )
    )


# =============================================================================
# I. Training, validation, and cross-scene evaluation
# =============================================================================

def train_yolo(model_name_or_path: str | Path, data_yaml: Path, project: Path, name: str, args: argparse.Namespace) -> Path:
    run_dir = project / name
    if args.clean_runs and run_dir.exists():
        clear_dir(run_dir)
    model = YOLO(str(model_name_or_path))
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.max_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        cache=parse_train_cache(args.train_cache),
        workers=0,
        project=str(project),
        name=name,
        exist_ok=True,
        seed=args.seed,
        amp=args.amp,
    )
    if args.ordinal_loss_gain > 0:
        from ordinal_detection_trainer import make_ordinal_trainer

        train_kwargs["trainer"] = make_ordinal_trainer(args.ordinal_loss_gain)
    if args.enable_source_aware_loss_mask:
        if "trainer" in train_kwargs:
            raise ValueError("--enable-source-aware-loss-mask cannot be combined with --ordinal-loss-gain yet.")
        from source_aware_detection_trainer import make_source_aware_trainer

        train_kwargs["trainer"] = make_source_aware_trainer()
    model.train(**train_kwargs)
    best_path = project / name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_path}")
    return best_path


def evaluate_model(model_path: Path, data_yaml: Path, split: str, args: argparse.Namespace):
    model = YOLO(str(model_path))
    return model.val(data=str(data_yaml), split=split, imgsz=args.imgsz, device=args.device, workers=0, verbose=False)


def load_label_boxes(label_path: Path, allowed_class_ids: Optional[Set[int]] = None) -> List[BoxPrediction]:
    boxes: List[BoxPrediction] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if allowed_class_ids is not None and class_id not in allowed_class_ids:
            continue
        if class_id not in CANONICAL_CLASSES:
            continue
        boxes.append(BoxPrediction(class_id, values[0], values[1], values[2], values[3], confidence=1.0))
    return boxes


def precision_recall_ap(tp_flags: Sequence[int], fp_flags: Sequence[int], scores: Sequence[float], total_gt: int) -> Tuple[float, float, float]:
    if total_gt <= 0:
        return 0.0, 0.0, 0.0
    ordered = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for idx in ordered:
        tp_cum += tp_flags[idx]
        fp_cum += fp_flags[idx]
        precision = tp_cum / max(tp_cum + fp_cum, 1)
        recall = tp_cum / total_gt
        precisions.append(precision)
        recalls.append(recall)
    if not precisions:
        return 0.0, 0.0, 0.0

    ap = 0.0
    for threshold in [step / 100 for step in range(0, 101)]:
        precision_at_recall = [p for p, r in zip(precisions, recalls) if r >= threshold]
        ap += (max(precision_at_recall) if precision_at_recall else 0.0) / 101.0
    return precisions[-1], recalls[-1], ap


def fixed_confidence_pr(
    gt_by_image: Dict[str, List[BoxPrediction]],
    pred_by_image: Dict[str, List[BoxPrediction]],
    class_id: int,
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> Tuple[float, float, float, int, int, int]:
    total_gt = sum(1 for boxes in gt_by_image.values() for box in boxes if box.class_id == class_id)
    matched: Dict[str, Set[int]] = {image_name: set() for image_name in gt_by_image}
    prediction_items = []
    for image_name, boxes in pred_by_image.items():
        for box in boxes:
            if box.class_id == class_id and box.confidence >= conf_threshold:
                prediction_items.append((image_name, box))
    prediction_items.sort(key=lambda item: item[1].confidence, reverse=True)

    tp = 0
    fp = 0
    for image_name, pred_box in prediction_items:
        gt_boxes = [box for box in gt_by_image[image_name] if box.class_id == class_id]
        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched[image_name]:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched[image_name].add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = max(total_gt - tp, 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1, tp, fp, fn


def parse_float_list(text: str, fallback: Sequence[float]) -> List[float]:
    if not text:
        return list(fallback)
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def box_to_pixels(box: BoxPrediction, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0.0, box.x_center - box.width / 2) * image_width)
    y1 = int(max(0.0, box.y_center - box.height / 2) * image_height)
    x2 = int(min(1.0, box.x_center + box.width / 2) * image_width)
    y2 = int(min(1.0, box.y_center + box.height / 2) * image_height)
    return x1, y1, x2, y2


def prediction_match_iou(
    pred_box: BoxPrediction,
    gt_boxes: Sequence[BoxPrediction],
    class_id: int,
) -> float:
    best_iou = 0.0
    for gt_box in gt_boxes:
        if gt_box.class_id != class_id:
            continue
        best_iou = max(best_iou, box_iou(pred_box, gt_box))
    return best_iou


def write_pr_curve_table(
    output_csv: Path,
    gt_by_image: Dict[str, List[BoxPrediction]],
    pred_by_image: Dict[str, List[BoxPrediction]],
    allowed_class_ids: Sequence[int],
    model_path: Path,
    dataset_root: Path,
    iou_threshold: float = 0.50,
) -> None:
    """
    Save full confidence-ranked PR curve points for the masked external set.

    This is separate from the fixed-threshold calibration table: AP integrates
    over this ranked curve, while fixed thresholds support deployment-oriented
    precision/recall claims at conf=0.25/0.40/0.50.
    """
    rows: List[dict] = []
    for class_id in allowed_class_ids:
        total_gt = sum(1 for boxes in gt_by_image.values() for box in boxes if box.class_id == class_id)
        matched: Dict[str, Set[int]] = {image_name: set() for image_name in gt_by_image}
        prediction_items: List[Tuple[str, BoxPrediction]] = []
        for image_name, boxes in pred_by_image.items():
            for box in boxes:
                if box.class_id == class_id:
                    prediction_items.append((image_name, box))
        prediction_items.sort(key=lambda item: item[1].confidence, reverse=True)

        tp = 0
        fp = 0
        for rank, (image_name, pred_box) in enumerate(prediction_items, start=1):
            gt_boxes = [box for box in gt_by_image[image_name] if box.class_id == class_id]
            best_iou = 0.0
            best_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched[image_name]:
                    continue
                iou = box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx
            if best_iou >= iou_threshold and best_idx >= 0:
                matched[image_name].add(best_idx)
                tp += 1
                is_tp = 1
            else:
                fp += 1
                is_tp = 0
            precision = tp / max(tp + fp, 1)
            recall = tp / max(total_gt, 1)
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": CANONICAL_CLASSES[class_id],
                    "rank": rank,
                    "confidence": f"{pred_box.confidence:.6f}",
                    "precision": f"{precision:.6f}",
                    "recall": f"{recall:.6f}",
                    "tp": tp,
                    "fp": fp,
                    "is_tp": is_tp,
                    "best_iou": f"{best_iou:.6f}",
                    "model_path": str(model_path),
                    "dataset_root": str(dataset_root),
                }
            )
    write_rows_csv(output_csv, rows)
    print(f"Saved PR curve table: {output_csv}")


def write_confidence_calibration_table(
    output_csv: Path,
    gt_by_image: Dict[str, List[BoxPrediction]],
    pred_by_image: Dict[str, List[BoxPrediction]],
    allowed_class_ids: Sequence[int],
    thresholds: Sequence[float],
    model_path: Path,
    dataset_root: Path,
) -> dict:
    rows = []
    best_macro = {"threshold": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in thresholds:
        class_metrics = []
        for class_id in allowed_class_ids:
            precision, recall, f1, tp, fp, fn = fixed_confidence_pr(
                gt_by_image,
                pred_by_image,
                class_id=class_id,
                conf_threshold=threshold,
                iou_threshold=0.50,
            )
            class_metrics.append((precision, recall, f1))
            rows.append(
                {
                    "threshold": f"{threshold:.3f}",
                    "class_id": class_id,
                    "class_name": CANONICAL_CLASSES[class_id],
                    "precision": f"{precision:.6f}",
                    "recall": f"{recall:.6f}",
                    "f1": f"{f1:.6f}",
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "model_path": str(model_path),
                    "dataset_root": str(dataset_root),
                }
            )
        macro_precision = sum(item[0] for item in class_metrics) / max(len(class_metrics), 1)
        macro_recall = sum(item[1] for item in class_metrics) / max(len(class_metrics), 1)
        macro_f1 = sum(item[2] for item in class_metrics) / max(len(class_metrics), 1)
        rows.append(
            {
                "threshold": f"{threshold:.3f}",
                "class_id": "",
                "class_name": "macro",
                "precision": f"{macro_precision:.6f}",
                "recall": f"{macro_recall:.6f}",
                "f1": f"{macro_f1:.6f}",
                "tp": "",
                "fp": "",
                "fn": "",
                "model_path": str(model_path),
                "dataset_root": str(dataset_root),
            }
        )
        if macro_f1 > best_macro["f1"]:
            best_macro = {
                "threshold": threshold,
                "f1": macro_f1,
                "precision": macro_precision,
                "recall": macro_recall,
            }
    write_rows_csv(output_csv, rows)
    print(f"Saved confidence calibration table: {output_csv}")
    return best_macro


def write_classwise_threshold_search_table(
    output_csv: Path,
    gt_by_image: Dict[str, List[BoxPrediction]],
    pred_by_image: Dict[str, List[BoxPrediction]],
    allowed_class_ids: Sequence[int],
    thresholds: Sequence[float],
    model_path: Path,
    dataset_root: Path,
    target_precision: float,
) -> dict:
    """
    Search a separate confidence threshold for each external-domain class.

    The primary selection maximizes recall among thresholds meeting target_precision.
    If no threshold reaches that precision for a class, the fallback is max F1.
    This makes the operating point explicit for precision-sensitive deployment
    while keeping recall loss visible in the table.
    """
    rows: List[dict] = []
    selected_metrics: List[dict] = []
    for class_id in allowed_class_ids:
        candidates: List[dict] = []
        for threshold in thresholds:
            precision, recall, f1, tp, fp, fn = fixed_confidence_pr(
                gt_by_image,
                pred_by_image,
                class_id=class_id,
                conf_threshold=threshold,
                iou_threshold=0.50,
            )
            candidate = {
                "threshold": threshold,
                "class_id": class_id,
                "class_name": CANONICAL_CLASSES[class_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "meets_target_precision": precision >= target_precision,
            }
            candidates.append(candidate)
            rows.append(
                {
                    **candidate,
                    "threshold": f"{threshold:.3f}",
                    "precision": f"{precision:.6f}",
                    "recall": f"{recall:.6f}",
                    "f1": f"{f1:.6f}",
                    "target_precision": f"{target_precision:.3f}",
                    "selected": 0,
                    "selection_rule": "",
                    "model_path": str(model_path),
                    "dataset_root": str(dataset_root),
                }
            )
        feasible = [item for item in candidates if item["meets_target_precision"]]
        if feasible:
            selected = max(feasible, key=lambda item: (item["recall"], item["f1"], -item["threshold"]))
            selection_rule = f"max_recall_with_precision>={target_precision:.2f}"
        else:
            selected = max(candidates, key=lambda item: (item["f1"], item["precision"], -item["threshold"]))
            selection_rule = "fallback_max_f1_no_threshold_met_target_precision"
        selected_metrics.append({**selected, "selection_rule": selection_rule})

    for row in rows:
        selected = next(item for item in selected_metrics if item["class_id"] == row["class_id"])
        if abs(float(row["threshold"]) - selected["threshold"]) < 1e-9:
            row["selected"] = 1
            row["selection_rule"] = selected["selection_rule"]

    macro_precision = sum(item["precision"] for item in selected_metrics) / max(len(selected_metrics), 1)
    macro_recall = sum(item["recall"] for item in selected_metrics) / max(len(selected_metrics), 1)
    macro_f1 = sum(item["f1"] for item in selected_metrics) / max(len(selected_metrics), 1)
    rows.append(
        {
            "threshold": "classwise",
            "class_id": "",
            "class_name": "macro_selected",
            "precision": f"{macro_precision:.6f}",
            "recall": f"{macro_recall:.6f}",
            "f1": f"{macro_f1:.6f}",
            "tp": sum(int(item["tp"]) for item in selected_metrics),
            "fp": sum(int(item["fp"]) for item in selected_metrics),
            "fn": sum(int(item["fn"]) for item in selected_metrics),
            "meets_target_precision": macro_precision >= target_precision,
            "target_precision": f"{target_precision:.3f}",
            "selected": 1,
            "selection_rule": "macro_of_selected_class_thresholds",
            "model_path": str(model_path),
            "dataset_root": str(dataset_root),
        }
    )
    write_rows_csv(output_csv, rows)
    print(f"Saved class-wise threshold search table: {output_csv}")
    return {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "target_precision": target_precision,
        "selected_thresholds": {CANONICAL_CLASSES[item["class_id"]]: item["threshold"] for item in selected_metrics},
        "csv": str(output_csv),
    }


def write_false_positive_examples(
    output_dir: Path,
    output_csv: Path,
    dataset_root: Path,
    split: str,
    gt_by_image: Dict[str, List[BoxPrediction]],
    pred_by_image: Dict[str, List[BoxPrediction]],
    allowed_class_ids: Sequence[int],
    conf_threshold: float,
    max_examples: int,
    iou_threshold: float = 0.50,
) -> None:
    """
    Save high-confidence false-positive examples for qualitative error analysis.

    Red boxes are false-positive predictions; green boxes are matching-class
    ground truth boxes. These images make the low raw precision issue auditable
    instead of hidden behind a calibrated F1 number.
    """
    if max_examples <= 0:
        return
    ensure_dir(output_dir)
    rows: List[dict] = []
    fp_items: List[Tuple[float, str, BoxPrediction, float]] = []
    allowed = set(allowed_class_ids)
    for image_name, boxes in pred_by_image.items():
        gt_boxes = gt_by_image.get(image_name, [])
        for pred_box in boxes:
            if pred_box.class_id not in allowed or pred_box.confidence < conf_threshold:
                continue
            best_iou = prediction_match_iou(pred_box, gt_boxes, pred_box.class_id)
            if best_iou < iou_threshold:
                fp_items.append((pred_box.confidence, image_name, pred_box, best_iou))
    fp_items.sort(key=lambda item: item[0], reverse=True)

    images_dir = dataset_root / split / "images"
    for index, (confidence, image_name, pred_box, best_iou) in enumerate(fp_items[:max_examples], start=1):
        image_path = images_dir / image_name
        if not image_path.exists():
            continue
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for gt_box in gt_by_image.get(image_name, []):
                if gt_box.class_id == pred_box.class_id:
                    draw.rectangle(box_to_pixels(gt_box, width, height), outline=(0, 180, 0), width=3)
            pred_pixels = box_to_pixels(pred_box, width, height)
            draw.rectangle(pred_pixels, outline=(220, 30, 30), width=3)
            label = f"FP {CANONICAL_CLASSES[pred_box.class_id]} {confidence:.2f} IoU={best_iou:.2f}"
            draw.rectangle((pred_pixels[0], max(0, pred_pixels[1] - 16), pred_pixels[0] + 260, pred_pixels[1]), fill=(220, 30, 30))
            draw.text((pred_pixels[0] + 3, max(0, pred_pixels[1] - 15)), label, fill=(255, 255, 255))
            out_name = f"fp_{index:03d}_{CANONICAL_CLASSES[pred_box.class_id]}_{Path(image_name).stem}.jpg"
            out_path = output_dir / out_name
            image.save(out_path, quality=92)
        rows.append(
            {
                "rank": index,
                "image": image_name,
                "example_path": str(out_path),
                "predicted_class": CANONICAL_CLASSES[pred_box.class_id],
                "confidence": f"{confidence:.6f}",
                "best_same_class_iou": f"{best_iou:.6f}",
                "threshold": f"{conf_threshold:.3f}",
            }
        )
    write_rows_csv(output_csv, rows)
    print(f"Saved false-positive examples: {output_dir}")


def evaluate_model_masked_classes(
    model_path: Path,
    dataset_root: Path,
    split: str,
    allowed_class_ids: Sequence[int],
    args: argparse.Namespace,
    output_csv: Optional[Path] = None,
    calibration_csv: Optional[Path] = None,
):
    """
    Custom external evaluator for partial-label datasets.

    Ultralytics' standard val path assumes all model classes are part of the
    target label space. preprocessed-ffb lacks empty and overripe annotations,
    so this evaluator filters both GT and predictions to the four annotated
    classes and ignores classes outside the mask.
    """
    allowed = set(allowed_class_ids)
    images = list_images(dataset_root / split / "images")
    labels_dir = dataset_root / split / "labels"
    model = YOLO(str(model_path))
    iou_thresholds = [0.50 + 0.05 * idx for idx in range(10)]
    class_rows = []

    gt_by_image: Dict[str, List[BoxPrediction]] = {
        image_path.name: load_label_boxes(labels_dir / f"{image_path.stem}.txt", allowed)
        for image_path in images
    }
    pred_by_image: Dict[str, List[BoxPrediction]] = {}
    for image_path in images:
        predictions = predict_one(model, image_path, imgsz=args.imgsz, device=args.device, conf=0.001)
        pred_by_image[image_path.name] = [box for box in predictions if box.class_id in allowed]

    ap50_values = []
    ap5095_values = []
    precision_values = []
    recall_values = []
    for class_id in allowed_class_ids:
        total_gt = sum(1 for boxes in gt_by_image.values() for box in boxes if box.class_id == class_id)
        per_threshold_ap = []
        pr_at_50 = (0.0, 0.0)
        for iou_threshold in iou_thresholds:
            tp_flags: List[int] = []
            fp_flags: List[int] = []
            scores: List[float] = []
            matched: Dict[str, Set[int]] = {image_name: set() for image_name in gt_by_image}
            prediction_items = []
            for image_name, boxes in pred_by_image.items():
                for box in boxes:
                    if box.class_id == class_id:
                        prediction_items.append((image_name, box))
            prediction_items.sort(key=lambda item: item[1].confidence, reverse=True)
            for image_name, pred_box in prediction_items:
                gt_boxes = [box for box in gt_by_image[image_name] if box.class_id == class_id]
                best_iou = 0.0
                best_idx = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched[image_name]:
                        continue
                    iou = box_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gt_idx
                if best_iou >= iou_threshold and best_idx >= 0:
                    matched[image_name].add(best_idx)
                    tp_flags.append(1)
                    fp_flags.append(0)
                else:
                    tp_flags.append(0)
                    fp_flags.append(1)
                scores.append(pred_box.confidence)
            precision, recall, ap = precision_recall_ap(tp_flags, fp_flags, scores, total_gt)
            per_threshold_ap.append(ap)
            if abs(iou_threshold - 0.50) < 1e-9:
                pr_at_50 = (precision, recall)
        ap50 = per_threshold_ap[0] if per_threshold_ap else 0.0
        ap5095 = sum(per_threshold_ap) / max(len(per_threshold_ap), 1)
        ap50_values.append(ap50)
        ap5095_values.append(ap5095)
        precision_values.append(pr_at_50[0])
        recall_values.append(pr_at_50[1])
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": CANONICAL_CLASSES[class_id],
                "gt_boxes": total_gt,
                "precision_at_50": f"{pr_at_50[0]:.6f}",
                "recall_at_50": f"{pr_at_50[1]:.6f}",
                "ap50": f"{ap50:.6f}",
                "ap50_95": f"{ap5095:.6f}",
            }
        )

    summary = {
        "stage": "external_preprocessed_ffb_masked_4class",
        "evaluated_classes": ",".join(PREPROCESSED_FFB_EVAL_CLASSES),
        "ignored_classes": "empty,overripe",
        "images": len(images),
        "precision": sum(precision_values) / max(len(precision_values), 1),
        "recall": sum(recall_values) / max(len(recall_values), 1),
        "map50": sum(ap50_values) / max(len(ap50_values), 1),
        "map5095": sum(ap5095_values) / max(len(ap5095_values), 1),
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
    }
    thresholds = parse_float_list(
        getattr(args, "calibration_thresholds", DEFAULT_CALIBRATION_THRESHOLDS),
        fallback=[0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    )
    if (
        calibration_csv is None
        and output_csv is not None
        and not getattr(args, "skip_threshold_search", False)
    ):
        calibration_csv = output_csv.with_name(f"{output_csv.stem}_calibration.csv")
    if calibration_csv is not None:
        best_calibration = write_confidence_calibration_table(
            calibration_csv,
            gt_by_image=gt_by_image,
            pred_by_image=pred_by_image,
            allowed_class_ids=allowed_class_ids,
            thresholds=thresholds,
            model_path=model_path,
            dataset_root=dataset_root,
        )
        summary["best_calibrated_threshold"] = best_calibration["threshold"]
        summary["best_calibrated_precision"] = best_calibration["precision"]
        summary["best_calibrated_recall"] = best_calibration["recall"]
        summary["best_calibrated_f1"] = best_calibration["f1"]
        summary["calibration_csv"] = str(calibration_csv)
    if output_csv is not None and not getattr(args, "skip_threshold_search", False):
        fixed_thresholds = parse_float_list(
            getattr(args, "fixed_eval_thresholds", DEFAULT_FIXED_EVAL_THRESHOLDS),
            fallback=[0.25, 0.40, 0.50],
        )
        fixed_csv = output_csv.with_name(f"{output_csv.stem}_fixed_thresholds.csv")
        write_confidence_calibration_table(
            fixed_csv,
            gt_by_image=gt_by_image,
            pred_by_image=pred_by_image,
            allowed_class_ids=allowed_class_ids,
            thresholds=fixed_thresholds,
            model_path=model_path,
            dataset_root=dataset_root,
        )
        classwise_thresholds = parse_float_list(
            getattr(args, "classwise_threshold_grid", DEFAULT_CLASSWISE_THRESHOLD_GRID),
            fallback=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        )
        classwise_csv = output_csv.with_name(f"{output_csv.stem}_classwise_threshold_search.csv")
        classwise_summary = write_classwise_threshold_search_table(
            classwise_csv,
            gt_by_image=gt_by_image,
            pred_by_image=pred_by_image,
            allowed_class_ids=allowed_class_ids,
            thresholds=classwise_thresholds,
            model_path=model_path,
            dataset_root=dataset_root,
            target_precision=float(getattr(args, "target_operating_precision", 0.78)),
        )
        pr_curve_csv = output_csv.with_name(f"{output_csv.stem}_pr_curve.csv")
        write_pr_curve_table(
            pr_curve_csv,
            gt_by_image=gt_by_image,
            pred_by_image=pred_by_image,
            allowed_class_ids=allowed_class_ids,
            model_path=model_path,
            dataset_root=dataset_root,
        )
        fp_dir = output_csv.with_name(f"{output_csv.stem}_false_positive_examples")
        fp_csv = output_csv.with_name(f"{output_csv.stem}_false_positive_examples.csv")
        write_false_positive_examples(
            fp_dir,
            fp_csv,
            dataset_root=dataset_root,
            split=split,
            gt_by_image=gt_by_image,
            pred_by_image=pred_by_image,
            allowed_class_ids=allowed_class_ids,
            conf_threshold=float(getattr(args, "false_positive_threshold", 0.40)),
            max_examples=int(getattr(args, "max_false_positive_examples", 24)),
        )
        summary["fixed_threshold_csv"] = str(fixed_csv)
        summary["classwise_threshold_search_csv"] = str(classwise_csv)
        summary["classwise_threshold_precision"] = classwise_summary["precision"]
        summary["classwise_threshold_recall"] = classwise_summary["recall"]
        summary["classwise_threshold_f1"] = classwise_summary["f1"]
        summary["classwise_thresholds"] = classwise_summary["selected_thresholds"]
        summary["pr_curve_csv"] = str(pr_curve_csv)
        summary["false_positive_examples_csv"] = str(fp_csv)
        summary["false_positive_examples_dir"] = str(fp_dir)
    if output_csv:
        rows = [
            {
                **summary,
                "class_id": "",
                "class_name": "macro",
                "gt_boxes": sum(int(row["gt_boxes"]) for row in class_rows),
                "precision_at_50": f"{summary['precision']:.6f}",
                "recall_at_50": f"{summary['recall']:.6f}",
                "ap50": f"{summary['map50']:.6f}",
                "ap50_95": f"{summary['map5095']:.6f}",
            },
            *class_rows,
        ]
        write_rows_csv(output_csv, rows)
        print(f"Saved masked external evaluation: {output_csv}")
    return summary


def group_unlabeled_scenes(new_images: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for image_path in list_images(new_images):
        scene = infer_scene(image_path, new_images)
        groups.setdefault(scene, []).append(image_path)
    return dict(sorted(groups.items()))


def sample_scene_images(image_paths: Sequence[Path], quota: int, seed: int) -> List[Path]:
    images = list(image_paths)
    if quota <= 0 or len(images) <= quota:
        return images
    rng = random.Random(seed)
    rng.shuffle(images)
    return sorted(images[:quota])


def default_teacher_model() -> Optional[Path]:
    candidates = [
        Path(r"LOCAL_PROJECT_ROOT\exp1-6\weights\best.pt"),
        Path(r"LOCAL_PROJECT_ROOT\exp1-7\weights\best.pt"),
        Path(r"LOCAL_PROJECT_ROOT\exp-6\weights\best.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def build_scene_test_dataset(scene: str, image_paths: Sequence[Path], image_root: Path, output_root: Path) -> Optional[Path]:
    """
    Build a temporary scene-specific YOLO test dataset when labels are available.
    """
    scene_root = output_root / "scene_eval" / scene
    clear_dir(scene_root)
    for split in ("train", "valid", "test"):
        ensure_dir(scene_root / split / "images")
        ensure_dir(scene_root / split / "labels")

    copied = 0
    for image_path in image_paths:
        label_path = None
        sibling = image_path.with_suffix(".txt")
        if sibling.exists():
            label_path = sibling
        else:
            for part_idx, part in enumerate(image_path.parts):
                if part.lower() == "images":
                    parts = list(image_path.parts)
                    parts[part_idx] = "labels"
                    candidate = Path(*parts).with_suffix(".txt")
                    if candidate.exists():
                        label_path = candidate
                        break
        if label_path is None:
            continue
        valid, _, _ = count_label_file(label_path)
        if valid == 0:
            continue
        copy_file(image_path, scene_root / "test" / "images" / image_path.name)
        copy_file(label_path, scene_root / "test" / "labels" / f"{image_path.stem}.txt")
        copied += 1

    if copied == 0:
        return None
    return write_data_yaml(scene_root)


def evaluate_external_scenes(model_path: Path, new_images: Path, output_root: Path, args: argparse.Namespace) -> None:
    groups = group_unlabeled_scenes(new_images)
    if not groups:
        print("No external scenes available for scene-level evaluation.")
        return
    print("\nCross-scene evaluation:")
    for scene, images in groups.items():
        data_yaml = build_scene_test_dataset(scene, images, new_images, output_root)
        if data_yaml is None:
            print(f"  {scene}: no labels found, mAP skipped; images={len(images)}")
            continue
        metrics = evaluate_model(model_path, data_yaml, split="test", args=args)
        print(
            f"  {scene}: P={metrics.box.mp:.4f}, R={metrics.box.mr:.4f}, "
            f"mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}"
        )


def save_summary_table(rows: Sequence[StageMetrics], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["stage", "precision", "recall", "map50", "map5095", "model_path", "data_yaml"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stage": row.stage,
                    "precision": f"{row.precision:.6f}",
                    "recall": f"{row.recall:.6f}",
                    "map50": f"{row.map50:.6f}",
                    "map5095": f"{row.map5095:.6f}",
                    "model_path": row.model_path,
                    "data_yaml": row.data_yaml,
                }
            )
    print(f"Saved summary table: {output_csv}")


def report_path(args: argparse.Namespace, output_root: Path, filename: str) -> Path:
    if getattr(args, "paper_layout", True):
        return output_root / "reports" / filename
    return output_root / filename


def dataset_artifact_root(args: argparse.Namespace, output_root: Path) -> Path:
    return output_root / "datasets" if getattr(args, "paper_layout", True) else output_root


def weights_artifact_root(args: argparse.Namespace, output_root: Path) -> Path:
    return output_root / "weights" if getattr(args, "paper_layout", True) else output_root


def copy_named_weight(src: Path, output_root: Path, filename: str, args: argparse.Namespace) -> Optional[Path]:
    if not getattr(args, "paper_layout", True) or not src.exists():
        return None
    dst = weights_artifact_root(args, output_root) / filename
    if src.resolve() == dst.resolve():
        return dst
    copy_file(src, dst)
    return dst


# =============================================================================
# J. Main orchestration
# =============================================================================

def prepare_supervised_sources(args: argparse.Namespace, output_root: Path, logger: Optional[JsonRunLogger] = None) -> Path:
    """
    Prepare the supervised labeled YOLO dataset.
    """
    if args.prepared_supervised_root:
        prepared = Path(args.prepared_supervised_root)
        if not (prepared / "data.yaml").exists():
            raise FileNotFoundError(f"--prepared-supervised-root must contain data.yaml: {prepared}")
        print(f"Reusing explicit prepared supervised dataset: {prepared}")
        if logger:
            logger.log("dataset_reused", source="prepared_supervised_root", data_yaml=prepared / "data.yaml")
        return prepared

    prepared_root = dataset_artifact_root(args, output_root) / "prepared_supervised"
    merged_root = dataset_artifact_root(args, output_root) / "prepared_supervised_merged"
    if args.reuse_prepared_dataset:
        if (merged_root / "data.yaml").exists():
            print(f"Reusing prepared merged supervised dataset: {merged_root}")
            if logger:
                logger.log("dataset_reused", source="prepared_supervised_merged", data_yaml=merged_root / "data.yaml")
            return merged_root
        local_original = prepared_root / "local_original"
        if (local_original / "data.yaml").exists() and not args.extra_supervised_dataset:
            print(f"Reusing prepared supervised dataset: {local_original}")
            if logger:
                logger.log("dataset_reused", source="local_original", data_yaml=local_original / "data.yaml")
            return local_original
    else:
        clear_dir(prepared_root)

    supervised_yamls: List[Path] = []
    if args.original_dataset:
        yaml_path = convert_to_yolo_format(Path(args.original_dataset), prepared_root / "local_original")
        if args.scene_aware_original_split:
            yaml_path = scene_aware_yolo_split(
                yaml_path.parent,
                prepared_root / "local_original_scene_aware",
                val_groups_arg=args.scene_val_groups,
                test_groups_arg=args.scene_test_groups,
                logger=logger,
            )
        supervised_yamls.append(yaml_path)
        if logger:
            logger.log("dataset_prepared", source="local_original", data_yaml=yaml_path)

    if args.download_open_data:
        download_root = output_root / "downloads"
        extracted_dirs = try_download_public_datasets(args, download_root, logger=logger)
        for idx, (source_name, extracted) in enumerate(extracted_dirs, start=1):
            try:
                yaml_path = convert_to_yolo_format(extracted, prepared_root / f"{source_name}_{idx}")
                supervised_yamls.append(yaml_path)
                if logger:
                    logger.log("dataset_prepared", source=source_name, extracted=extracted, data_yaml=yaml_path)
            except Exception as exc:
                print(f"WARNING: Could not convert downloaded dataset {extracted}: {exc}")
                if logger:
                    logger.log("dataset_conversion_failed", source=source_name, extracted=extracted, error=str(exc))

    if args.roboflow_zip_url:
        extracted = download_and_extract(args.roboflow_zip_url, output_root / "downloads" / "roboflow", filename="roboflow_export.zip")
        yaml_path = convert_to_yolo_format(extracted, prepared_root / "roboflow")
        supervised_yamls.append(yaml_path)
        if logger:
            logger.log("dataset_prepared", source="roboflow", extracted=extracted, data_yaml=yaml_path)

    for idx, dataset_path in enumerate(args.extra_supervised_dataset or [], start=1):
        dataset_root = Path(dataset_path)
        if not dataset_root.exists():
            print(f"WARNING: extra supervised dataset does not exist: {dataset_root}")
            if logger:
                logger.log("extra_supervised_missing", path=dataset_root)
            continue
        try:
            yaml_path = convert_to_yolo_format(dataset_root, prepared_root / f"extra_supervised_{idx}")
            supervised_yamls.append(yaml_path)
            if logger:
                logger.log("dataset_prepared", source=f"extra_supervised_{idx}", extracted=dataset_root, data_yaml=yaml_path)
        except Exception as exc:
            print(f"WARNING: Could not convert extra supervised dataset {dataset_root}: {exc}")
            if logger:
                logger.log("dataset_conversion_failed", source=f"extra_supervised_{idx}", extracted=dataset_root, error=str(exc))

    if not supervised_yamls:
        raise ValueError("No supervised dataset was prepared. Pass --original-dataset and/or --download-open-data.")

    if len(supervised_yamls) == 1:
        return supervised_yamls[0].parent

    clear_dir(merged_root)
    source_roots = []
    for idx, data_yaml in enumerate(supervised_yamls, start=1):
        source_root = data_yaml.parent
        source_roots.append(source_root)
        for split in ("train", "valid", "test"):
            copy_yolo_split(source_root, split, merged_root, prefix=f"src{idx}_")
    write_data_yaml(merged_root)
    merge_source_metadata(source_roots, merged_root)
    inspect_yolo_dataset(merged_root)
    return merged_root


def generate_strict_evaluation_protocol_reports(
    args: argparse.Namespace,
    output_root: Path,
    logger: Optional[JsonRunLogger] = None,
) -> dict:
    """
    Generate SCI-style evaluation protocol artifacts before any training starts.

    Random-split results are written only as in-domain references. The main
    conclusion should use scene-disjoint ScienceDB/OILPALM and masked external
    preprocessed-ffb evaluation.
    """
    reports_root = output_root / "reports" if getattr(args, "paper_layout", True) else output_root
    prepared_root = dataset_artifact_root(args, output_root) / "prepared_supervised"
    protocol_root = dataset_artifact_root(args, output_root) / "strict_protocol"
    ensure_dir(protocol_root)
    report = {
        "random_split_csv": str(reports_root / "baseline_random_split.csv"),
        "scene_disjoint_csv": str(reports_root / "baseline_scene_disjoint.csv"),
        "external_preprocessed_ffb_csv": str(reports_root / "external_preprocessed_ffb.csv"),
        "pseudo_label_ablation_csv": str(reports_root / "pseudo_label_ablation.csv"),
    }

    original_root = prepared_root / "local_original"
    if original_root.exists():
        write_dataset_protocol_table(
            original_root,
            reports_root / "baseline_random_split.csv",
            protocol_role="in_domain_reference_only_random_or_official_split_not_main_conclusion",
            main_conclusion=False,
        )
        scene_root = protocol_root / "science_oilpalm_scene_disjoint"
        try:
            scene_yaml = scene_aware_yolo_split(
                original_root,
                scene_root,
                val_groups_arg=args.scene_val_groups,
                test_groups_arg=args.scene_test_groups,
                logger=logger,
            )
            write_dataset_protocol_table(
                scene_yaml.parent,
                reports_root / "baseline_scene_disjoint.csv",
                protocol_role="main_in_domain_scene_disjoint_protocol",
                main_conclusion=True,
            )
            report["scene_disjoint_data_yaml"] = str(scene_yaml)
        except Exception as exc:
            write_rows_csv(
                reports_root / "baseline_scene_disjoint.csv",
                [{"error": str(exc), "protocol_role": "main_in_domain_scene_disjoint_protocol_failed"}],
            )
            report["scene_disjoint_error"] = str(exc)
    else:
        write_rows_csv(reports_root / "baseline_random_split.csv", [{"error": "local_original_missing"}])
        write_rows_csv(reports_root / "baseline_scene_disjoint.csv", [{"error": "local_original_missing"}])
        report["warning"] = "No local_original dataset was prepared, so ScienceDB/OILPALM split reports are empty."

    external_yaml: Optional[Path] = None
    if args.external_test_dataset:
        external_yaml = convert_to_yolo_format(Path(args.external_test_dataset), protocol_root / "external_preprocessed_ffb_canonical")
    elif args.extra_supervised_dataset:
        external_yaml = convert_to_yolo_format(Path(args.extra_supervised_dataset[-1]), protocol_root / "external_preprocessed_ffb_canonical")

    masked_external_root: Optional[Path] = None
    if external_yaml is not None:
        masked_yaml = create_masked_external_dataset(
            external_yaml.parent,
            protocol_root / "external_preprocessed_ffb_masked_4class",
            PREPROCESSED_FFB_EVAL_CLASS_IDS,
        )
        masked_external_root = masked_yaml.parent
        write_dataset_protocol_table(
            masked_external_root,
            reports_root / "external_preprocessed_ffb.csv",
            protocol_role="external_domain_masked_4class_empty_overripe_ignored",
            main_conclusion=True,
        )
        report["external_masked_data_yaml"] = str(masked_yaml)
    else:
        write_rows_csv(reports_root / "external_preprocessed_ffb.csv", [{"error": "external_preprocessed_ffb_missing"}])
        report["external_warning"] = "No --external-test-dataset or --extra-supervised-dataset was available for preprocessed-ffb protocol reporting."

    duplicate_sources: List[Tuple[str, Path]] = []
    if original_root.exists():
        duplicate_sources.append(("science_oilpalm_original", original_root))
    if masked_external_root is not None:
        duplicate_sources.append(("preprocessed_ffb_masked_external", masked_external_root))
    if duplicate_sources:
        duplicate_summary = write_duplicate_report(
            duplicate_sources,
            reports_root / "duplicate_report.csv",
            use_phash=args.enable_phash_duplicate_check,
            phash_threshold=args.phash_hamming_threshold,
        )
        report["duplicate_check"] = duplicate_summary

    write_pseudo_ablation_table(reports_root / "pseudo_label_ablation.csv")
    protocol_json = reports_root / "strict_evaluation_protocol.json"
    protocol_json.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    if logger:
        logger.log("strict_evaluation_protocol_generated", **report)
    print(f"Saved strict evaluation protocol summary: {protocol_json}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Full YOLOv8 SSOD pipeline for oil palm FFB maturity detection.")
    parser.add_argument("--original-dataset", type=str, default=r"LOCAL_DATA_ROOT\OILPALM.yolov8\split_dataset")
    parser.add_argument("--new-images", type=str, default=r"LOCAL_DATA_ROOT\new_images")
    parser.add_argument("--external-test-dataset", type=str, default="", help="Optional labeled external YOLO/COCO/class-folder dataset used only for final generalization testing.")
    parser.add_argument("--prepared-external-masked-root", type=str, default="", help="Reuse an existing masked external YOLO dataset root containing data.yaml.")
    parser.add_argument("--teacher-model", type=str, default="", help="Optional strong teacher best.pt for pseudo-labeling. If omitted, the script auto-detects a local exp*/best.pt when available.")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Ultralytics model used for supervised baseline training, e.g. yolov8n.pt, yolov8s.pt, yolo11n.pt, yolo11s.pt, rtdetr-l.pt.")
    parser.add_argument("--baseline-model-path", type=str, default="", help="Existing trained detector to evaluate as the supervised baseline.")
    parser.add_argument("--skip-baseline-training", action="store_true", help="Skip supervised training and use --baseline-model-path or the auto-detected local best.pt.")
    parser.add_argument("--output-root", type=str, default=r"LOCAL_PROJECT_ROOT\full_ssod_pipeline")
    parser.add_argument("--prepared-supervised-root", type=str, default="", help="Reuse an existing prepared YOLO dataset root containing data.yaml.")
    parser.add_argument("--reuse-prepared-dataset", action="store_true", help="Reuse prepared datasets inside --output-root when data.yaml already exists.")
    parser.add_argument("--skip-protocol-reports", action="store_true", help="Skip random/scene/external/duplicate protocol CSV generation for faster iterative runs.")
    parser.add_argument("--paper-layout", action=argparse.BooleanOptionalAction, default=True, help="Write outputs under datasets/, reports/, runs/, and weights/ inside --output-root.")
    parser.add_argument("--download-open-data", action="store_true", help="Download open Zenodo FFB data and add it to supervised training.")
    parser.add_argument("--roboflow-zip-url", type=str, default="", help="Optional direct Roboflow YOLOv8 export zip URL.")
    parser.add_argument("--extra-dataset-url", action="append", default=[], help="Optional direct archive URL for Science Data Bank or other public mirrors. Can be repeated.")
    parser.add_argument("--extra-supervised-dataset", action="append", default=[], help="Additional local labeled dataset in YOLO/COCO/class-folder format to merge into supervised training. Can be repeated.")
    parser.add_argument("--scene-aware-original-split", action="store_true", help="Pool the original dataset and rebuild train/valid/test by non-overlapping video/source groups.")
    parser.add_argument("--scene-val-groups", type=str, default="", help="Comma-separated source groups for scene-aware validation. Empty chooses automatically.")
    parser.add_argument("--scene-test-groups", type=str, default="", help="Comma-separated source groups for scene-aware test. Empty chooses automatically.")
    parser.add_argument("--protocol-report-only", action="store_true", help="Generate strict split, duplicate, external-mask, and ablation reports, then exit before training.")
    parser.add_argument("--enable-phash-duplicate-check", action="store_true", help="Also scan cross-dataset perceptual-hash near duplicates. Slower on large image sets.")
    parser.add_argument("--phash-hamming-threshold", type=int, default=5, help="Hamming-distance threshold for optional perceptual-hash duplicate candidates.")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--self-training-iterations", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--train-cache", type=str, default="none", help="Ultralytics cache setting: none, ram, or disk.")
    parser.add_argument("--ordinal-loss-gain", type=float, default=0.0, help="Optional ordered-maturity auxiliary classification loss gain.")
    parser.add_argument("--enable-source-aware-loss-mask", action="store_true", help="Actively mask classification BCE gradients for classes absent from each merged labeled source.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-thresholds", type=str, default=DEFAULT_CALIBRATION_THRESHOLDS, help="Comma-separated confidence thresholds for masked external calibration tables.")
    parser.add_argument("--fixed-eval-thresholds", type=str, default=DEFAULT_FIXED_EVAL_THRESHOLDS, help="Deployment-style fixed thresholds reported separately, e.g. 0.25,0.40,0.50.")
    parser.add_argument("--classwise-threshold-grid", type=str, default=DEFAULT_CLASSWISE_THRESHOLD_GRID, help="Grid for per-class external threshold search.")
    parser.add_argument("--target-operating-precision", type=float, default=0.85, help="Preferred minimum per-class precision for class-wise threshold selection. The selected operating point maximizes recall while meeting this target when possible.")
    parser.add_argument("--false-positive-threshold", type=float, default=0.40, help="Confidence threshold used to export qualitative false-positive examples.")
    parser.add_argument("--max-false-positive-examples", type=int, default=24, help="Maximum qualitative false-positive images saved per masked external evaluation.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP. Leave off if Ultralytics AMP check hits a bad local yolo*.pt file.")
    parser.add_argument("--quality-threshold", type=float, default=0.72)
    parser.add_argument("--quality-threshold-by-class", type=str, default="", help="Class-specific Q thresholds, e.g. abnormal=0.62,under_ripe=0.62.")
    parser.add_argument("--base-conf-threshold", type=float, default=0.55)
    parser.add_argument("--min-conf-threshold", type=float, default=0.40)
    parser.add_argument("--max-conf-threshold", type=float, default=0.85)
    parser.add_argument("--class-threshold-overrides", type=str, default="", help="Class-specific confidence thresholds, e.g. abnormal=0.45,under_ripe=0.45.")
    parser.add_argument("--target-map", type=float, default=0.90)
    parser.add_argument("--consistency-iou", type=float, default=0.60)
    parser.add_argument("--min-consistency-support", type=int, default=2)
    parser.add_argument("--num-aug-views", type=int, default=4)
    parser.add_argument(
        "--consistency-profile",
        choices=("at_weak_strong", "mic_masked"),
        default="at_weak_strong",
        help="Pseudo-label consistency views. mic_masked adds coordinate-preserving patch masking inspired by MIC.",
    )
    parser.add_argument("--use-vertical-flip", action="store_true")
    parser.add_argument("--min-box-area", type=float, default=0.0005)
    parser.add_argument("--max-box-area", type=float, default=0.70)
    parser.add_argument("--max-aspect-ratio", type=float, default=6.0)
    parser.add_argument("--edge-margin", type=float, default=0.002)
    parser.add_argument("--reject-edge-contact", action="store_true", help="Precision-first SSOD: reject pseudo boxes touching the configured edge margin.")
    parser.add_argument("--max-boxes-per-image", type=int, default=40)
    parser.add_argument("--max-pseudo-images-per-class", type=int, default=0, help="Per iteration/scene cap for accepted pseudo-label images per class. <=0 disables the cap.")
    parser.add_argument("--max-pseudo-boxes-per-class", type=int, default=0, help="Per iteration/scene cap for accepted pseudo-label boxes per class. <=0 disables the default cap.")
    parser.add_argument("--max-pseudo-boxes-per-class-overrides", type=str, default="", help="Class-specific box caps, e.g. abnormal=80,under_ripe=80,ripe=150.")
    parser.add_argument("--scene-quota-per-iteration", type=int, default=250, help="Max unlabeled images sampled from each scene per iteration. <=0 uses all images.")
    parser.add_argument("--held-out-scene", type=str, default="", help="Optional scene name to reserve from pseudo-label training when no external test dataset is supplied.")
    parser.add_argument("--disable-consistency", action="store_true", help="Ablation: accept pseudo labels without multi-view consistency support.")
    parser.add_argument("--disable-quality-scoring", action="store_true", help="Ablation: accept pseudo labels without composite Q filtering.")
    parser.add_argument("--disable-dynamic-thresholds", action="store_true", help="Ablation: use the base confidence threshold for every class.")
    parser.add_argument("--allow-folder-mismatch", action="store_true", help="Ablation: do not reject pseudo labels whose predicted class conflicts with folder metadata.")
    parser.add_argument("--enable-folder-guided-hard-class-relabel", action="store_true", help="Weak-label-guided SSOD: relabel high-quality adjacent-class predictions to the image-folder weak class for hard classes.")
    parser.add_argument("--hard-classes", type=str, default="abnormal,under_ripe")
    parser.add_argument("--hard-class-source-map", type=str, default="under_ripe:unripe|ripe,abnormal:empty|unripe|overripe")
    parser.add_argument("--hard-class-min-quality", type=float, default=0.80)
    parser.add_argument("--hard-class-min-conf", type=float, default=0.55)
    parser.add_argument("--hard-class-min-support", type=int, default=1)
    parser.add_argument("--ensemble-teacher-model", action="append", default=[], help="Optional additional detector best.pt used only for pseudo-label voting. Can be repeated.")
    parser.add_argument("--min-ensemble-support", type=int, default=2, help="Minimum total teacher votes including the primary teacher. Ignored when no ensemble teachers are supplied.")
    parser.add_argument("--ensemble-iou", type=float, default=0.55)
    parser.add_argument("--enable-ordinal-under-ripe-rescue", action="store_true", help="Weak-label ordinal rescue: conservatively relabel adjacent ripe/unripe predictions inside under_ripe folders when multi-view and ensemble checks pass.")
    parser.add_argument("--ordinal-min-quality", type=float, default=0.72)
    parser.add_argument("--ordinal-min-conf", type=float, default=0.45)
    parser.add_argument("--ordinal-min-view-support", type=int, default=2)
    parser.add_argument("--ordinal-min-ensemble-support", type=int, default=2)
    parser.add_argument("--self-train-from-teacher", action=argparse.BooleanOptionalAction, default=True, help="Initialize each self-training student from the current teacher instead of plain yolov8n.pt.")
    parser.add_argument("--clean-runs", action=argparse.BooleanOptionalAction, default=True, help="Clear matching run directories before training to avoid stale partial results.")
    args = parser.parse_args()

    random.seed(args.seed)
    args.class_threshold_overrides = parse_class_float_map(args.class_threshold_overrides)
    args.quality_threshold_by_class = parse_class_float_map(args.quality_threshold_by_class)
    args.max_pseudo_boxes_per_class_overrides = parse_class_int_map(args.max_pseudo_boxes_per_class_overrides)
    args.hard_class_ids = parse_class_list(args.hard_classes)
    args.hard_class_source_map = parse_hard_class_source_map(args.hard_class_source_map)
    args.ensemble_teacher_models = [Path(path) for path in args.ensemble_teacher_model]
    missing_ensemble_models = [path for path in args.ensemble_teacher_models if not path.exists()]
    if missing_ensemble_models:
        raise FileNotFoundError(f"Missing --ensemble-teacher-model paths: {missing_ensemble_models}")
    output_root = Path(args.output_root)
    ensure_dir(output_root)
    set_safe_delete_root(output_root)
    if args.paper_layout:
        for child in ("datasets", "reports", "runs", "weights"):
            ensure_dir(output_root / child)
    runs_root = output_root / "runs"
    ensure_dir(runs_root)
    logger = JsonRunLogger(report_path(args, output_root, "structured_results.json"))
    logger.log("run_started", args=vars(args), open_dataset_sources=OPEN_DATASET_SOURCES)
    if args.enable_folder_guided_hard_class_relabel:
        logger.log(
            "weak_label_guided_ssod_enabled",
            method_name="weak-label-guided SSOD with folder-level image-class priors",
            weak_label_source="parent folder name is treated as an image-level weak label, not as bounding-box ground truth",
            hard_classes=[CANONICAL_CLASSES[class_id] for class_id in sorted(args.hard_class_ids)],
            hard_class_source_map={
                CANONICAL_CLASSES[target_id]: [CANONICAL_CLASSES[source_id] for source_id in sorted(source_ids)]
                for target_id, source_ids in args.hard_class_source_map.items()
            },
            paper_note=(
                "Report this setting as weak-label-guided self-training. It is not pure unlabeled SSOD, "
                "because folder labels provide image-level class priors for abnormal and under_ripe."
            ),
        )

    supervised_root = prepare_supervised_sources(args, output_root, logger=logger)
    supervised_data_yaml = write_data_yaml(supervised_root)
    supervised_summary = inspect_yolo_dataset(supervised_root)
    logger.log("dataset_inspected", dataset=supervised_root, summary=supervised_summary)
    if args.skip_protocol_reports:
        protocol_report = {"skipped": True, "reason": "--skip-protocol-reports"}
        logger.log("strict_evaluation_protocol_skipped", **protocol_report)
        print("Skipping strict protocol reports for this iterative run.")
    else:
        protocol_report = generate_strict_evaluation_protocol_reports(args, output_root, logger=logger)
    if args.protocol_report_only:
        logger.log("run_finished", reason="protocol_report_only", protocol_report=protocol_report)
        print("\nProtocol reports generated. Exiting before any training/evaluation run.")
        return

    if args.skip_baseline_training:
        baseline_candidate = Path(args.baseline_model_path) if args.baseline_model_path else default_teacher_model()
        if not baseline_candidate or not baseline_candidate.exists():
            raise FileNotFoundError("--skip-baseline-training requires --baseline-model-path or an auto-detected local best.pt")
        baseline_best = baseline_candidate
        print(f"\nSkipping supervised baseline training; using existing model: {baseline_best}")
    else:
        print(f"\nTraining supervised baseline from {args.base_model}...")
        baseline_best = train_yolo(args.base_model, supervised_data_yaml, runs_root, "baseline_supervised", args)
    copied_baseline = copy_named_weight(baseline_best, output_root, "baseline_best.pt", args)
    if copied_baseline:
        logger.log("weight_exported", role="baseline", source=baseline_best, exported=copied_baseline)
    baseline_metrics = evaluate_model(baseline_best, supervised_data_yaml, split="val", args=args)
    summary_rows = [metrics_to_stage("baseline_val", baseline_metrics, baseline_best, supervised_data_yaml)]
    logger.log("metrics", stage="baseline_val", metrics=summary_rows[-1], per_class=per_class_metric_summary(baseline_metrics))
    baseline_test_metrics = evaluate_model(baseline_best, supervised_data_yaml, split="test", args=args)
    summary_rows.append(metrics_to_stage("baseline_test", baseline_test_metrics, baseline_best, supervised_data_yaml))
    logger.log("metrics", stage="baseline_test", metrics=summary_rows[-1], per_class=per_class_metric_summary(baseline_test_metrics))

    external_test_root: Optional[Path] = None
    external_test_yaml: Optional[Path] = None
    if args.prepared_external_masked_root:
        external_test_root = Path(args.prepared_external_masked_root)
        external_test_yaml = external_test_root / "data.yaml"
        if not external_test_yaml.exists():
            raise FileNotFoundError(f"--prepared-external-masked-root must contain data.yaml: {external_test_root}")
        print(f"Reusing prepared masked external dataset: {external_test_root}")
        logger.log(
            "external_test_reused",
            dataset=external_test_root,
            data_yaml=external_test_yaml,
            evaluated_classes=PREPROCESSED_FFB_EVAL_CLASSES,
            ignored_classes=["empty", "overripe"],
        )
    elif args.external_test_dataset:
        external_canonical_yaml = convert_to_yolo_format(Path(args.external_test_dataset), dataset_artifact_root(args, output_root) / "external_test_prepared")
        external_test_yaml = create_masked_external_dataset(
            external_canonical_yaml.parent,
            dataset_artifact_root(args, output_root) / "external_test_preprocessed_ffb_masked_4class",
            PREPROCESSED_FFB_EVAL_CLASS_IDS,
        )
        external_test_root = external_test_yaml.parent
        logger.log(
            "external_test_prepared",
            dataset=external_test_root,
            data_yaml=external_test_yaml,
            evaluated_classes=PREPROCESSED_FFB_EVAL_CLASSES,
            ignored_classes=["empty", "overripe"],
        )
    if external_test_root is not None and external_test_yaml is not None:
        baseline_external_summary = evaluate_model_masked_classes(
            baseline_best,
            external_test_root,
            split="test",
            allowed_class_ids=PREPROCESSED_FFB_EVAL_CLASS_IDS,
            args=args,
            output_csv=report_path(args, output_root, "external_preprocessed_ffb_metrics.csv"),
        )
        summary_rows.append(
            StageMetrics(
                stage="baseline_external_preprocessed_ffb_masked_4class",
                precision=float(baseline_external_summary["precision"]),
                recall=float(baseline_external_summary["recall"]),
                map50=float(baseline_external_summary["map50"]),
                map5095=float(baseline_external_summary["map5095"]),
                model_path=str(baseline_best),
                data_yaml=str(external_test_yaml),
            )
        )
        logger.log("metrics", stage="baseline_external_preprocessed_ffb_masked_4class", metrics=summary_rows[-1], masked_summary=baseline_external_summary)

    if args.disable_dynamic_thresholds:
        thresholds = {class_id: args.base_conf_threshold for class_id in CANONICAL_CLASSES}
    else:
        thresholds = dynamic_class_threshold_update(
            baseline_metrics,
            base_threshold=args.base_conf_threshold,
            min_threshold=args.min_conf_threshold,
            max_threshold=args.max_conf_threshold,
            target_map=args.target_map,
        )
    thresholds = apply_class_threshold_overrides(thresholds, args.class_threshold_overrides)
    print_thresholds(thresholds)
    logger.log("dynamic_thresholds", stage="baseline_val", thresholds=thresholds)

    teacher_path = Path(args.teacher_model) if args.teacher_model else default_teacher_model()
    if teacher_path and teacher_path.exists():
        pseudo_teacher_path = teacher_path
        teacher_metrics = evaluate_model(teacher_path, supervised_data_yaml, split="val", args=args)
        teacher_stage = metrics_to_stage("teacher_val", teacher_metrics, teacher_path, supervised_data_yaml)
        summary_rows.append(teacher_stage)
        logger.log("metrics", stage="teacher_val", metrics=teacher_stage, per_class=per_class_metric_summary(teacher_metrics))
        if not args.disable_dynamic_thresholds:
            thresholds = dynamic_class_threshold_update(
                teacher_metrics,
                base_threshold=args.base_conf_threshold,
                min_threshold=args.min_conf_threshold,
                max_threshold=args.max_conf_threshold,
                target_map=args.target_map,
            )
            thresholds = apply_class_threshold_overrides(thresholds, args.class_threshold_overrides)
            print_thresholds(thresholds)
            logger.log("dynamic_thresholds", stage="teacher_val", thresholds=thresholds)
        print(f"Using strong teacher for pseudo-labeling: {pseudo_teacher_path}")
    else:
        pseudo_teacher_path = baseline_best
        print(f"Using baseline as pseudo-label teacher: {pseudo_teacher_path}")

    new_images_root = Path(args.new_images)
    scene_groups = group_unlabeled_scenes(new_images_root) if new_images_root.exists() else {}
    if not scene_groups:
        print(f"No unlabeled images found in {new_images_root}; stopping after supervised baseline.")
        save_summary_table(summary_rows, report_path(args, output_root, "summary_metrics.csv"))
        logger.log("run_finished", final_model=baseline_best, reason="no_unlabeled_images")
        return

    scene_names = list(scene_groups)

    held_out_scene = args.held_out_scene.strip()
    if held_out_scene and held_out_scene not in scene_groups:
        raise ValueError(f"--held-out-scene '{held_out_scene}' was not found. Available scenes: {scene_names}")
    train_scenes = [scene for scene in scene_names if scene != held_out_scene]
    if not train_scenes:
        train_scenes = scene_names
    print(f"\nScene groups: { {scene: len(scene_groups[scene]) for scene in scene_names} }")
    if held_out_scene:
        print(f"Held-out scene reserved from pseudo-labeling: {held_out_scene}")
    if external_test_yaml:
        print(f"External test dataset: {external_test_yaml}")
    logger.log(
        "scene_groups",
        groups={scene: len(scene_groups[scene]) for scene in scene_names},
        held_out_scene=held_out_scene,
        train_scenes=train_scenes,
        external_test=external_test_yaml,
    )

    current_model_path = pseudo_teacher_path if args.self_training_iterations > 0 else baseline_best
    pseudo_sources: List[Tuple[Path, Path]] = []

    for iteration in range(1, args.self_training_iterations + 1):
        print(f"\nSelf-training iteration {iteration}: balanced scenes={train_scenes}")
        for scene in train_scenes:
            sampled_images = sample_scene_images(
                scene_groups[scene],
                quota=args.scene_quota_per_iteration,
                seed=args.seed + iteration * 1009 + len(scene),
            )
            pseudo_images, pseudo_labels, pseudo_results = generate_pseudo_labels_for_scene(
                model_path=current_model_path,
                image_paths=sampled_images,
                image_root=new_images_root,
                output_root=output_root / "pseudo",
                class_thresholds=thresholds,
                args=args,
                iteration=iteration,
                scene=scene,
            )
            pseudo_report = output_root / "pseudo" / f"iter_{iteration}_{scene}" / "pseudo_report.csv"
            logger.log(
                "pseudo_labels_generated",
                iteration=iteration,
                scene=scene,
                sampled_images=len(sampled_images),
                images=pseudo_images,
                labels=pseudo_labels,
                report=pseudo_report,
                summary=summarize_pseudo_results(pseudo_results),
            )
            pseudo_sources.append((pseudo_images, pseudo_labels))

        combined_root = output_root / f"combined_iter_{iteration}"
        if args.paper_layout:
            combined_root = dataset_artifact_root(args, output_root) / f"combined_iter_{iteration}"
        combined_yaml = assemble_combined_dataset(supervised_root, pseudo_sources, combined_root)
        combined_summary = inspect_yolo_dataset(combined_root)
        logger.log("combined_dataset", iteration=iteration, data_yaml=combined_yaml, metadata=combined_root / "source_metadata.json", summary=combined_summary)

        student_init = current_model_path if args.self_train_from_teacher else args.base_model
        current_model_path = train_yolo(student_init, combined_yaml, runs_root, f"self_train_iter_{iteration}", args)
        val_metrics = evaluate_model(current_model_path, combined_yaml, split="val", args=args)
        summary_rows.append(metrics_to_stage(f"iter_{iteration}_val", val_metrics, current_model_path, combined_yaml))
        logger.log("metrics", stage=f"iter_{iteration}_val", metrics=summary_rows[-1], per_class=per_class_metric_summary(val_metrics))

        if args.disable_dynamic_thresholds:
            thresholds = {class_id: args.base_conf_threshold for class_id in CANONICAL_CLASSES}
        else:
            thresholds = dynamic_class_threshold_update(
                val_metrics,
                base_threshold=args.base_conf_threshold,
                min_threshold=args.min_conf_threshold,
                max_threshold=args.max_conf_threshold,
                target_map=args.target_map,
            )
        thresholds = apply_class_threshold_overrides(thresholds, args.class_threshold_overrides)
        print_thresholds(thresholds)
        logger.log("dynamic_thresholds", stage=f"iter_{iteration}_val", thresholds=thresholds)

        heldout_yaml = None
        heldout_name = ""
        if external_test_yaml is not None:
            heldout_yaml = external_test_yaml
            heldout_name = "external_test"
        elif held_out_scene:
            heldout_yaml = build_scene_test_dataset(held_out_scene, scene_groups[held_out_scene], new_images_root, output_root)
            heldout_name = held_out_scene

        if heldout_yaml is not None:
            if external_test_root is not None and heldout_yaml == external_test_yaml:
                heldout_summary = evaluate_model_masked_classes(
                    current_model_path,
                    external_test_root,
                    split="test",
                    allowed_class_ids=PREPROCESSED_FFB_EVAL_CLASS_IDS,
                    args=args,
                    output_csv=report_path(args, output_root, f"iter_{iteration}_external_preprocessed_ffb_metrics.csv"),
                )
                summary_rows.append(
                    StageMetrics(
                        stage=f"iter_{iteration}_heldout_external_preprocessed_ffb_masked_4class",
                        precision=float(heldout_summary["precision"]),
                        recall=float(heldout_summary["recall"]),
                        map50=float(heldout_summary["map50"]),
                        map5095=float(heldout_summary["map5095"]),
                        model_path=str(current_model_path),
                        data_yaml=str(heldout_yaml),
                    )
                )
                logger.log("metrics", stage=summary_rows[-1].stage, metrics=summary_rows[-1], masked_summary=heldout_summary)
                print(
                    f"Held-out {heldout_name} masked 4-class: P={heldout_summary['precision']:.4f}, "
                    f"R={heldout_summary['recall']:.4f}, mAP50={heldout_summary['map50']:.4f}, "
                    f"mAP50-95={heldout_summary['map5095']:.4f}"
                )
            else:
                heldout_metrics = evaluate_model(current_model_path, heldout_yaml, split="test", args=args)
                summary_rows.append(metrics_to_stage(f"iter_{iteration}_heldout_{heldout_name}", heldout_metrics, current_model_path, heldout_yaml))
                logger.log("metrics", stage=f"iter_{iteration}_heldout_{heldout_name}", metrics=summary_rows[-1], per_class=per_class_metric_summary(heldout_metrics))
                print(
                    f"Held-out {heldout_name}: P={heldout_metrics.box.mp:.4f}, R={heldout_metrics.box.mr:.4f}, "
                    f"mAP50={heldout_metrics.box.map50:.4f}, mAP50-95={heldout_metrics.box.map:.4f}"
                )
        elif held_out_scene:
            print(f"Held-out scene {held_out_scene} has no labels, so mAP is skipped.")

    evaluate_external_scenes(current_model_path, new_images_root, output_root, args)
    summary_csv = report_path(args, output_root, "summary_metrics.csv")
    save_summary_table(summary_rows, summary_csv)
    exported_final = copy_named_weight(current_model_path, output_root, "final_best.pt", args) if args.self_training_iterations > 0 else None
    if exported_final:
        logger.log("weight_exported", role="final", source=current_model_path, exported=exported_final)
    logger.log("summary", rows=summary_rows, summary_csv=summary_csv)
    logger.log("run_finished", final_model=current_model_path)

    print("\nPipeline complete.")
    print(f"Final model: {current_model_path}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()

