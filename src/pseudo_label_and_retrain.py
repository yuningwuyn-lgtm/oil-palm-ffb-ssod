"""
Pseudo-label and retrain pipeline for YOLOv8 oil palm FFB detection.

How to use:
1. Edit paths and training settings in the Configuration section below.
2. Run this script directly in PyCharm.
3. Check pseudo_pipeline/combined_dataset labels before trusting training results.
4. Check pseudo_pipeline/runs output for retraining results and demo predictions.

Required package:
    pip install ultralytics
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ultralytics import YOLO


# =============================================================================
# A. Configuration
# =============================================================================

TRAINED_MODEL = Path(r"LOCAL_PROJECT_ROOT\exp-6\weights\best.pt")
ORIGINAL_DATASET = Path(r"LOCAL_DATA_ROOT\OILPALM.yolov8\split_dataset")
NEW_IMAGES = Path(r"LOCAL_DATA_ROOT\new_images")
OUTPUT_ROOT = Path(r"LOCAL_PROJECT_ROOT\pseudo_pipeline")

CONF_THRESHOLD = 0.60
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Pseudo-label quality gates. Tune these after reviewing rejected samples.
MIN_BOX_SIDE = 0.01
MIN_BOX_AREA = 0.0005
MAX_BOX_AREA = 0.70
MAX_ASPECT_RATIO = 6.0
EDGE_MARGIN = 0.002
MAX_BOXES_PER_IMAGE = 40
REJECT_IF_ANY_BAD_BOX = False
REJECT_IF_CLASS_DIFFERS_FROM_FOLDER = True

# Optional manual rejection list. Put one image filename or stem per line.
MANUAL_REJECT_FILE = OUTPUT_ROOT / "manual_reject_list.txt"

PSEUDO_TRAIN_RATIO = 0.8
PSEUDO_VAL_RATIO = 0.1
PSEUDO_SPLIT_SEED = 42

EPOCHS = 10
IMGSZ = 640
BATCH = 8
DEVICE = "0"  # Use "cpu" if CUDA is unavailable.

CLASS_NAMES = {
    0: "abnormal",
    1: "empty",
    2: "overripe",
    3: "ripe",
    4: "under_ripe",
    5: "unripe",
}

CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_NAMES.items()}

# Add project-specific folder aliases here if your folders use different names.
FOLDER_CLASS_ALIASES = {
    "under-ripe": "under_ripe",
    "underripe": "under_ripe",
    "under ripe": "under_ripe",
}


# =============================================================================
# B. Utility functions
# =============================================================================

def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    """Delete and recreate a directory."""
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def list_images(folder: Path) -> List[Path]:
    """Return image files in a folder recursively, sorted for reproducibility."""
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )


def copy_file(src: Path, dst: Path) -> None:
    """Copy one file, creating the destination directory if needed."""
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def load_manual_rejects(reject_file: Path) -> Set[str]:
    """Load manually rejected image filenames/stems."""
    if not reject_file.exists():
        return set()

    rejected: Set[str] = set()
    with reject_file.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.split("#", 1)[0].strip()
            if value:
                rejected.add(value.lower())

    print(f"Loaded manual rejects: {len(rejected)} from {reject_file}")
    return rejected


def is_manually_rejected(image_path: Path, rejected_names: Set[str]) -> bool:
    """Return True when the image filename or stem is in the manual reject list."""
    return (
        image_path.name.lower() in rejected_names
        or image_path.stem.lower() in rejected_names
    )


def normalize_folder_name(folder_name: str) -> str:
    """Normalize a folder name before matching it to a class name."""
    normalized = folder_name.strip().lower().replace("-", "_").replace(" ", "_")
    return FOLDER_CLASS_ALIASES.get(normalized, normalized)


def expected_class_from_folder(image_path: Path, image_root: Path) -> int | None:
    """
    Return the class implied by the image's direct parent folder.

    Images directly inside NEW_IMAGES have no folder label and skip this check.
    Unknown class folders return -1 so the image is rejected instead of silently
    becoming noisy training data.
    """
    try:
        relative_path = image_path.resolve().relative_to(image_root.resolve())
    except ValueError:
        relative_path = image_path

    if len(relative_path.parts) < 2:
        return None

    folder_class_name = normalize_folder_name(relative_path.parts[-2])
    return CLASS_NAME_TO_ID.get(folder_class_name, -1)


def validate_pseudo_box(
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    confidence: float,
    conf_threshold: float,
) -> List[str]:
    """Return rejection reasons for an unsafe pseudo-label box."""
    reasons: List[str] = []

    if class_id not in CLASS_NAMES:
        reasons.append("unknown_class")
    if confidence < conf_threshold:
        reasons.append("low_confidence")
    if width <= 0.0 or height <= 0.0:
        reasons.append("non_positive_box")
        return reasons
    if width < MIN_BOX_SIDE or height < MIN_BOX_SIDE:
        reasons.append("box_side_too_small")

    area = width * height
    if area < MIN_BOX_AREA:
        reasons.append("box_area_too_small")
    if area > MAX_BOX_AREA:
        reasons.append("box_area_too_large")

    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > MAX_ASPECT_RATIO:
        reasons.append("bad_aspect_ratio")

    x_min = x_center - width / 2.0
    y_min = y_center - height / 2.0
    x_max = x_center + width / 2.0
    y_max = y_center + height / 2.0
    if (
        x_min < EDGE_MARGIN
        or y_min < EDGE_MARGIN
        or x_max > 1.0 - EDGE_MARGIN
        or y_max > 1.0 - EDGE_MARGIN
    ):
        reasons.append("touches_image_edge")

    return reasons


def count_label_objects(label_file: Path) -> int:
    """
    Count valid YOLO detection rows in a label file.

    Expected format:
        class_id x_center y_center width height

    Values after class_id must be normalized between 0 and 1.
    """
    if not label_file.exists() or label_file.stat().st_size == 0:
        return 0

    valid_rows = 0
    with label_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 5:
                print(f"WARNING: Invalid column count in {label_file} line {line_number}: {stripped}")
                continue

            try:
                class_id = int(float(parts[0]))
                values = [float(value) for value in parts[1:]]
            except ValueError:
                print(f"WARNING: Non-numeric label in {label_file} line {line_number}: {stripped}")
                continue

            if class_id not in CLASS_NAMES:
                print(f"WARNING: Unknown class_id {class_id} in {label_file} line {line_number}")
                continue

            if not all(0.0 <= value <= 1.0 for value in values):
                print(f"WARNING: Non-normalized bbox in {label_file} line {line_number}: {stripped}")
                continue

            width = values[2]
            height = values[3]
            if width <= 0.0 or height <= 0.0:
                print(f"WARNING: Non-positive bbox size in {label_file} line {line_number}: {stripped}")
                continue

            valid_rows += 1

    return valid_rows


def inspect_yolo_dataset(dataset_root: Path) -> Dict[str, Dict[str, int]]:
    """Print and return basic image/label/object counts for train/valid/test splits."""
    print(f"\nInspecting YOLO dataset: {dataset_root}")
    summary: Dict[str, Dict[str, int]] = {}

    total_objects = 0
    total_label_files = 0

    for split in ("train", "valid", "test"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        images = list_images(images_dir)
        labels = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []

        empty_labels = 0
        valid_objects = 0
        for label_file in labels:
            object_count = count_label_objects(label_file)
            valid_objects += object_count
            if object_count == 0:
                empty_labels += 1

        missing_labels = 0
        for image_path in images:
            if not (labels_dir / f"{image_path.stem}.txt").exists():
                missing_labels += 1

        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "valid_objects": valid_objects,
            "empty_labels": empty_labels,
            "missing_labels": missing_labels,
        }

        total_objects += valid_objects
        total_label_files += len(labels)

        print(
            f"  {split}: images={len(images)}, labels={len(labels)}, "
            f"objects={valid_objects}, empty_labels={empty_labels}, "
            f"missing_labels={missing_labels}"
        )

    if total_label_files > 0 and total_objects == 0:
        print(
            "\nSTRONG WARNING: Original dataset labels are empty. "
            "This dataset cannot train YOLO detection unless labels are fixed."
        )

    return summary


def safe_write_yaml(data_yaml_path: Path, dataset_root: Path, class_names: Dict[int, str]) -> None:
    """Write a clean YOLOv8 data.yaml file with forward-slash paths."""
    ensure_dir(data_yaml_path.parent)
    dataset_path = dataset_root.resolve().as_posix()

    lines = [
        f"path: {dataset_path}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for class_id in sorted(class_names):
        lines.append(f"  {class_id}: {class_names[class_id]}")

    data_yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote data.yaml: {data_yaml_path}")


def delete_cache_files(dataset_root: Path) -> None:
    """Delete YOLO .cache files that may contain stale dataset metadata."""
    for cache_file in dataset_root.rglob("*.cache"):
        try:
            cache_file.unlink()
            print(f"Deleted cache file: {cache_file}")
        except OSError as exc:
            print(f"WARNING: Could not delete cache file {cache_file}: {exc}")


def unique_destination(base_dir: Path, source_path: Path, used_names: set[str]) -> Path:
    """
    Return a destination path that avoids overwriting files with the same name.

    YOLO requires image and label stems to match, so callers must use the same
    returned stem for the paired image and label file.
    """
    candidate_name = source_path.name
    candidate_stem = source_path.stem
    suffix = source_path.suffix
    counter = 1

    while candidate_name.lower() in used_names or (base_dir / candidate_name).exists():
        candidate_name = f"{candidate_stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate_name.lower())
    return base_dir / candidate_name


# =============================================================================
# C. Pseudo-label generation
# =============================================================================

def generate_pseudo_labels(
    model_path: Path,
    image_folder: Path,
    output_root: Path,
    conf_threshold: float,
) -> Tuple[Path, Path, int]:
    """
    Generate filtered pseudo-labels from a trained YOLO model.

    Labels are written directly from result.boxes as:
        class_id x_center y_center width height

    Empty label files are not written, and images with no valid detections are skipped.
    """
    raw_labels_dir = output_root / "pseudo_labels_raw"
    filtered_labels_dir = output_root / "pseudo_labels_filtered"
    pseudo_images_dir = output_root / "pseudo_labeled_images"
    rejected_images_dir = output_root / "pseudo_rejected_images"
    rejected_labels_dir = output_root / "pseudo_rejected_labels"
    review_report = output_root / "pseudo_label_review_report.csv"

    clear_dir(raw_labels_dir)
    clear_dir(filtered_labels_dir)
    clear_dir(pseudo_images_dir)
    clear_dir(rejected_images_dir)
    clear_dir(rejected_labels_dir)
    ensure_dir(output_root)

    images = list_images(image_folder)
    if not images:
        print(f"No images found in: {image_folder}")
        return pseudo_images_dir, filtered_labels_dir, 0

    print(f"\nLoading trained YOLO model: {model_path}")
    model = YOLO(model_path)

    image_list_file = output_root / "new_images_list.txt"
    image_list_file.write_text(
        "\n".join(image_path.resolve().as_posix() for image_path in images) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Predicting {len(images)} images from: {image_folder}")
    results = model.predict(
        # Pass a recursive image-list file instead of a Python list. Ultralytics
        # uses PIL to pre-load Python list inputs, which can fail on malformed
        # EXIF metadata in otherwise readable images.
        source=str(image_list_file),
        conf=0.25,
        save=False,
        save_txt=False,
        save_conf=False,
        device=DEVICE,
    )

    accepted_count = 0
    rejected_count = 0
    used_image_names: set[str] = set()
    used_rejected_names: set[str] = set()
    manual_rejects = load_manual_rejects(MANUAL_REJECT_FILE)
    report_rows: List[Dict[str, str]] = []

    for result in results:
        image_path = Path(result.path)
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        xywhn = boxes.xywhn.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        raw_lines: List[str] = []
        filtered_lines: List[str] = []
        rejected_lines: List[str] = []
        image_reasons: List[str] = []
        expected_class_id = (
            expected_class_from_folder(image_path, image_folder)
            if REJECT_IF_CLASS_DIFFERS_FROM_FOLDER
            else None
        )

        for box, class_value, confidence in zip(xywhn, classes, confidences):
            class_id = int(class_value)
            x_center, y_center, width, height = box

            # Clamp tiny numeric overflow from model output before validation.
            x_center = min(max(float(x_center), 0.0), 1.0)
            y_center = min(max(float(y_center), 0.0), 1.0)
            width = min(max(float(width), 0.0), 1.0)
            height = min(max(float(height), 0.0), 1.0)
            confidence = float(confidence)

            label_without_conf = (
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}"
            )
            label_with_conf = f"{label_without_conf} {confidence:.6f}"
            raw_lines.append(label_with_conf)

            reasons = validate_pseudo_box(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                confidence=confidence,
                conf_threshold=conf_threshold,
            )
            if expected_class_id == -1:
                reasons.append("unknown_folder_class")
            elif expected_class_id is not None and class_id != expected_class_id:
                reasons.append("folder_class_mismatch")

            if reasons:
                rejected_lines.append(f"{label_with_conf} {'|'.join(reasons)}")
                image_reasons.extend(reasons)
            else:
                filtered_lines.append(label_without_conf)

        if raw_lines:
            raw_label_path = raw_labels_dir / f"{image_path.stem}.txt"
            raw_label_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8", newline="\n")

        if len(filtered_lines) > MAX_BOXES_PER_IMAGE:
            image_reasons.append("too_many_boxes")
            rejected_lines.extend(f"{line} too_many_boxes" for line in filtered_lines)
            filtered_lines = []

        if REJECT_IF_ANY_BAD_BOX and rejected_lines:
            image_reasons.append("contains_rejected_box")
            rejected_lines.extend(f"{line} contains_rejected_box" for line in filtered_lines)
            filtered_lines = []

        if is_manually_rejected(image_path, manual_rejects):
            image_reasons.append("manual_reject")
            rejected_lines.extend(f"{line} manual_reject" for line in filtered_lines)
            filtered_lines = []

        if not filtered_lines:
            if raw_lines or rejected_lines:
                destination_image = unique_destination(rejected_images_dir, image_path, used_rejected_names)
                destination_label = rejected_labels_dir / f"{destination_image.stem}.txt"
                copy_file(image_path, destination_image)
                destination_label.write_text(
                    "\n".join(rejected_lines or raw_lines) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                rejected_count += 1
                report_rows.append(
                    {
                        "image": str(image_path),
                        "status": "rejected",
                        "expected_class": CLASS_NAMES.get(expected_class_id, "unknown")
                        if expected_class_id is not None
                        else "",
                        "accepted_boxes": "0",
                        "rejected_boxes": str(len(rejected_lines)),
                        "reasons": "|".join(sorted(set(image_reasons))) or "no_valid_box",
                    }
                )
            continue

        destination_image = unique_destination(pseudo_images_dir, image_path, used_image_names)
        destination_label = filtered_labels_dir / f"{destination_image.stem}.txt"

        copy_file(image_path, destination_image)
        destination_label.write_text(
            "\n".join(filtered_lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        accepted_count += 1
        report_rows.append(
            {
                "image": str(image_path),
                "status": "accepted",
                "expected_class": CLASS_NAMES.get(expected_class_id, "unknown")
                if expected_class_id is not None
                else "",
                "accepted_boxes": str(len(filtered_lines)),
                "rejected_boxes": str(len(rejected_lines)),
                "reasons": "|".join(sorted(set(image_reasons))),
            }
        )

    with review_report.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "status",
                "expected_class",
                "accepted_boxes",
                "rejected_boxes",
                "reasons",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Accepted pseudo-labeled images: {accepted_count}")
    print(f"Rejected pseudo-labeled images: {rejected_count}")
    print(f"Raw labels with confidence: {raw_labels_dir}")
    print(f"Filtered training labels without confidence: {filtered_labels_dir}")
    print(f"Accepted images: {pseudo_images_dir}")
    print(f"Rejected images for review: {rejected_images_dir}")
    print(f"Rejected labels with reasons: {rejected_labels_dir}")
    print(f"Review report: {review_report}")

    return pseudo_images_dir, filtered_labels_dir, accepted_count


# =============================================================================
# D. Combine datasets
# =============================================================================

def copy_labeled_split(
    source_images: Path,
    source_labels: Path,
    destination_images: Path,
    destination_labels: Path,
    prefix: str = "",
) -> int:
    """Copy images only when a matching non-empty valid label exists."""
    ensure_dir(destination_images)
    ensure_dir(destination_labels)

    copied = 0
    used_names = {path.name.lower() for path in destination_images.iterdir() if path.is_file()}

    for image_path in list_images(source_images):
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"WARNING: Skipping image without label: {image_path}")
            continue

        if label_path.stat().st_size == 0:
            print(f"WARNING: Skipping image with 0KB label: {image_path}")
            continue

        if count_label_objects(label_path) == 0:
            print(f"WARNING: Skipping image with no valid YOLO objects: {image_path}")
            continue

        if prefix:
            candidate_image = destination_images / f"{prefix}{image_path.name}"
            counter = 1
            while candidate_image.name.lower() in used_names or candidate_image.exists():
                candidate_image = destination_images / f"{prefix}{image_path.stem}_{counter}{image_path.suffix}"
                counter += 1
            used_names.add(candidate_image.name.lower())
        else:
            candidate_image = unique_destination(destination_images, image_path, used_names)

        candidate_label = destination_labels / f"{candidate_image.stem}.txt"
        copy_file(image_path, candidate_image)
        copy_file(label_path, candidate_label)
        copied += 1

    return copied


def split_labeled_images(
    source_images: Path,
    source_labels: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Path]]:
    """Return valid pseudo-labeled images split into train/valid/test."""
    valid_images: List[Path] = []

    for image_path in list_images(source_images):
        label_path = source_labels / f"{image_path.stem}.txt"
        if (
            label_path.exists()
            and label_path.stat().st_size > 0
            and count_label_objects(label_path) > 0
        ):
            valid_images.append(image_path)

    rng = random.Random(seed)
    rng.shuffle(valid_images)

    total = len(valid_images)
    train_cut = int(total * train_ratio)
    val_cut = int(total * (train_ratio + val_ratio))

    return {
        "train": valid_images[:train_cut],
        "valid": valid_images[train_cut:val_cut],
        "test": valid_images[val_cut:],
    }


def copy_labeled_files(
    image_paths: List[Path],
    source_labels: Path,
    destination_images: Path,
    destination_labels: Path,
    prefix: str,
) -> int:
    """Copy a provided list of labeled images into one YOLO split."""
    ensure_dir(destination_images)
    ensure_dir(destination_labels)

    copied = 0
    used_names = {path.name.lower() for path in destination_images.iterdir() if path.is_file()}

    for image_path in image_paths:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists() or count_label_objects(label_path) == 0:
            print(f"WARNING: Skipping invalid pseudo image during split copy: {image_path}")
            continue

        candidate_image = destination_images / f"{prefix}{image_path.name}"
        counter = 1
        while candidate_image.name.lower() in used_names or candidate_image.exists():
            candidate_image = destination_images / f"{prefix}{image_path.stem}_{counter}{image_path.suffix}"
            counter += 1
        used_names.add(candidate_image.name.lower())

        candidate_label = destination_labels / f"{candidate_image.stem}.txt"
        copy_file(image_path, candidate_image)
        copy_file(label_path, candidate_label)
        copied += 1

    return copied


def build_combined_dataset(
    original_dataset: Path,
    pseudo_images: Path,
    pseudo_labels: Path,
    combined_root: Path,
) -> None:
    """
    Build a YOLOv8 combined dataset.

    Original train/valid/test are copied into matching splits.
    Pseudo-labeled images are split into train/valid/test and added to each split.
    """
    clear_dir(combined_root)

    split_dirs = [
        combined_root / "train" / "images",
        combined_root / "train" / "labels",
        combined_root / "valid" / "images",
        combined_root / "valid" / "labels",
        combined_root / "test" / "images",
        combined_root / "test" / "labels",
    ]
    for directory in split_dirs:
        ensure_dir(directory)

    train_images = combined_root / "train" / "images"
    train_labels = combined_root / "train" / "labels"
    valid_images = combined_root / "valid" / "images"
    valid_labels = combined_root / "valid" / "labels"
    test_images = combined_root / "test" / "images"
    test_labels = combined_root / "test" / "labels"

    copied_original_train = copy_labeled_split(
        original_dataset / "train" / "images",
        original_dataset / "train" / "labels",
        train_images,
        train_labels,
    )

    copied_original_valid = copy_labeled_split(
        original_dataset / "valid" / "images",
        original_dataset / "valid" / "labels",
        valid_images,
        valid_labels,
    )

    copied_original_test = copy_labeled_split(
        original_dataset / "test" / "images",
        original_dataset / "test" / "labels",
        test_images,
        test_labels,
    )

    pseudo_splits = split_labeled_images(
        source_images=pseudo_images,
        source_labels=pseudo_labels,
        train_ratio=PSEUDO_TRAIN_RATIO,
        val_ratio=PSEUDO_VAL_RATIO,
        seed=PSEUDO_SPLIT_SEED,
    )

    copied_pseudo_train = copy_labeled_files(
        pseudo_splits["train"],
        pseudo_labels,
        train_images,
        train_labels,
        prefix="pseudo_",
    )
    copied_pseudo_valid = copy_labeled_files(
        pseudo_splits["valid"],
        pseudo_labels,
        valid_images,
        valid_labels,
        prefix="pseudo_",
    )
    copied_pseudo_test = copy_labeled_files(
        pseudo_splits["test"],
        pseudo_labels,
        test_images,
        test_labels,
        prefix="pseudo_",
    )

    delete_cache_files(combined_root)

    print("\nCombined dataset copied:")
    print(f"  original train images: {copied_original_train}")
    print(f"  original valid images: {copied_original_valid}")
    print(f"  original test images:  {copied_original_test}")
    print(f"  pseudo train images:   {copied_pseudo_train}")
    print(f"  pseudo valid images:   {copied_pseudo_valid}")
    print(f"  pseudo test images:    {copied_pseudo_test}")
    print(f"  total train images:    {copied_original_train + copied_pseudo_train}")
    print(f"  total valid images:    {copied_original_valid + copied_pseudo_valid}")
    print(f"  total test images:     {copied_original_test + copied_pseudo_test}")


# =============================================================================
# F. Retrain YOLO
# =============================================================================

def train_yolo(data_yaml: Path, output_project: Path) -> YOLO:
    """Retrain YOLOv8 on the combined dataset."""
    print(f"\nStarting YOLOv8 training with data: {data_yaml}")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=0,
        project=str(output_project),
        name="pseudo_retrain_exp",
        exist_ok=True,
    )
    return model


# =============================================================================
# G. Prediction demo
# =============================================================================

def run_prediction_demo(model: YOLO, valid_images_dir: Path, output_project: Path) -> None:
    """Run a demo prediction pass on validation images after training."""
    if not list_images(valid_images_dir):
        print(f"Skipping demo predictions because no validation images were found: {valid_images_dir}")
        return

    print(f"\nSaving demo predictions from: {valid_images_dir}")
    model.predict(
        source=str(valid_images_dir),
        conf=0.25,
        save=True,
        project=str(output_project),
        name="demo_predictions",
        exist_ok=True,
    )


# =============================================================================
# H. Main function
# =============================================================================

def main() -> None:
    raw_labels_dir = OUTPUT_ROOT / "pseudo_labels_raw"
    filtered_labels_dir = OUTPUT_ROOT / "pseudo_labels_filtered"
    pseudo_images_dir = OUTPUT_ROOT / "pseudo_labeled_images"
    combined_root = OUTPUT_ROOT / "combined_dataset"
    data_yaml = combined_root / "data.yaml"
    output_project = OUTPUT_ROOT / "runs"

    print("Pseudo-label and retrain configuration:")
    print(f"  TRAINED_MODEL:    {TRAINED_MODEL}")
    print(f"  ORIGINAL_DATASET: {ORIGINAL_DATASET}")
    print(f"  NEW_IMAGES:       {NEW_IMAGES}")
    print(f"  OUTPUT_ROOT:      {OUTPUT_ROOT}")
    print(f"  CONF_THRESHOLD:   {CONF_THRESHOLD}")
    print(f"  MIN_BOX_SIDE:     {MIN_BOX_SIDE}")
    print(f"  MIN_BOX_AREA:     {MIN_BOX_AREA}")
    print(f"  MAX_BOX_AREA:     {MAX_BOX_AREA}")
    print(f"  MAX_ASPECT_RATIO: {MAX_ASPECT_RATIO}")
    print(f"  EDGE_MARGIN:      {EDGE_MARGIN}")
    print(f"  MAX_BOXES/IMAGE:  {MAX_BOXES_PER_IMAGE}")
    print(f"  FOLDER_CLASS_CHECK: {REJECT_IF_CLASS_DIFFERS_FROM_FOLDER}")
    print(f"  MANUAL_REJECTS:   {MANUAL_REJECT_FILE}")
    print(f"  PSEUDO_SPLIT:     train={PSEUDO_TRAIN_RATIO}, valid={PSEUDO_VAL_RATIO}, test={1.0 - PSEUDO_TRAIN_RATIO - PSEUDO_VAL_RATIO}")
    print(f"  PSEUDO_SPLIT_SEED:{PSEUDO_SPLIT_SEED}")
    print(f"  EPOCHS:           {EPOCHS}")
    print(f"  IMGSZ:            {IMGSZ}")
    print(f"  BATCH:            {BATCH}")
    print(f"  DEVICE:           {DEVICE}")
    print(f"  RAW_LABELS:       {raw_labels_dir}")
    print(f"  FILTERED_LABELS:  {filtered_labels_dir}")
    print(f"  PSEUDO_IMAGES:    {pseudo_images_dir}")
    print(f"  COMBINED_DATASET: {combined_root}")

    if not TRAINED_MODEL.exists():
        raise FileNotFoundError(f"Trained model not found: {TRAINED_MODEL}")
    if not ORIGINAL_DATASET.exists():
        raise FileNotFoundError(f"Original dataset not found: {ORIGINAL_DATASET}")
    if not NEW_IMAGES.exists():
        raise FileNotFoundError(f"New image folder not found: {NEW_IMAGES}")

    original_summary = inspect_yolo_dataset(ORIGINAL_DATASET)
    original_objects = sum(split["valid_objects"] for split in original_summary.values())
    original_label_files = sum(split["labels"] for split in original_summary.values())
    if original_label_files > 0 and original_objects == 0:
        print(
            "\nOriginal dataset labels are empty. "
            "This dataset cannot train YOLO detection unless labels are fixed."
        )

    pseudo_images, pseudo_labels, accepted_count = generate_pseudo_labels(
        model_path=TRAINED_MODEL,
        image_folder=NEW_IMAGES,
        output_root=OUTPUT_ROOT,
        conf_threshold=CONF_THRESHOLD,
    )

    print(f"\nPseudo-labeled images accepted: {accepted_count}")
    if accepted_count == 0:
        print(
            "Stopping before retraining because no pseudo-labeled images were accepted. "
            "The confidence threshold may be too high, the trained model may not be suitable, "
            "or the new image folder may not contain detectable oil palm FFB objects."
        )
        return

    build_combined_dataset(
        original_dataset=ORIGINAL_DATASET,
        pseudo_images=pseudo_images,
        pseudo_labels=pseudo_labels,
        combined_root=combined_root,
    )

    safe_write_yaml(data_yaml, combined_root, CLASS_NAMES)
    inspect_yolo_dataset(combined_root)

    trained_model = train_yolo(data_yaml, output_project)

    valid_images_dir = combined_root / "valid" / "images"
    run_prediction_demo(trained_model, valid_images_dir, output_project)

    final_model_path = output_project / "pseudo_retrain_exp" / "weights" / "best.pt"
    print(f"\nFinal model path: {final_model_path}")
    print(f"Combined dataset path: {combined_root}")
    print(f"Data YAML path: {data_yaml}")


if __name__ == "__main__":
    main()

