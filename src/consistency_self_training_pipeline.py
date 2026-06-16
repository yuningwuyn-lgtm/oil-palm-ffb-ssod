"""
Consistency self-training pipeline for YOLOv8 oil palm FFB maturity detection.

Run this file directly in PyCharm after editing the Configuration section.

Pipeline:
1. Evaluate the baseline model and derive dynamic class-wise thresholds.
2. Predict pseudo-labels on NEW_IMAGES.
3. Filter pseudo-labels with augmentation consistency and composite quality scoring.
4. Add selected pseudo-labels scene-by-scene into a combined YOLO dataset.
5. Retrain YOLOv8n on the combined dataset.
6. Save demo predictions and report scene-level generalization metrics when labels exist.
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageEnhance
from ultralytics import YOLO


# =============================================================================
# A. Configuration
# =============================================================================

# Required inputs.
TRAINED_MODEL = Path(r"LOCAL_PROJECT_ROOT\exp1-6\weights\best.pt")
ORIGINAL_DATASET = Path(r"LOCAL_DATA_ROOT\OILPALM.yolov8\split_dataset")
NEW_IMAGES = Path(r"LOCAL_DATA_ROOT\new_images")

# Output root. Existing generated pseudo-label folders are recreated each run.
OUTPUT_ROOT = Path(r"LOCAL_PROJECT_ROOT\consistency_self_training")

# YOLO training settings.
YOLO_BASE_MODEL = "yolov8n.pt"
DEVICE = "0"  # Use "cpu" if CUDA is unavailable.
EPOCHS = 10
IMGSZ = 640
BATCH = 8
WORKERS = 0  # Keep 0 for Windows/PyCharm stability.
SEED = 42

# Class names must match the original data.yaml.
CLASS_NAMES = {
    0: "abnormal",
    1: "empty",
    2: "overripe",
    3: "ripe",
    4: "under_ripe",
    5: "unripe",
}
CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_NAMES.items()}
FOLDER_CLASS_ALIASES = {
    "under-ripe": "under_ripe",
    "underripe": "under_ripe",
    "under ripe": "under_ripe",
}

# Baseline pseudo-label settings.
BASE_CONF_THRESHOLD = 0.55
MIN_CONF_THRESHOLD = 0.40
MAX_CONF_THRESHOLD = 0.85
TARGET_CLASS_MAP = 0.90
THRESHOLD_ADJUSTMENT_STRENGTH = 0.25

# Composite quality scoring settings.
QUALITY_THRESHOLD = 0.68
MIN_BOX_AREA = 0.0005
IDEAL_MIN_BOX_AREA = 0.0030
IDEAL_MAX_BOX_AREA = 0.2500
MAX_BOX_AREA = 0.7000
MAX_ASPECT_RATIO = 6.0
EDGE_MARGIN = 0.002
REJECT_IF_FOLDER_CLASS_MISMATCH = True
MAX_BOXES_PER_IMAGE = 40

# Consistency training settings.
CONSISTENCY_IOU_THRESHOLD = 0.60
MIN_AUGMENTATION_SUPPORT = 2
CONSISTENCY_AUGMENTATIONS = (
    "brightness_up",
    "brightness_down",
    "contrast_up",
    "horizontal_flip",
    "rotate90",
)

# Cross-scene self-training settings.
# A scene is inferred from the first folder under NEW_IMAGES. Images directly in
# NEW_IMAGES are assigned to "root_scene".
SELF_TRAINING_ITERATIONS = 1
PSEUDO_TRAIN_RATIO = 0.80
PSEUDO_VAL_RATIO = 0.10
MAX_ACCEPTED_IMAGES_PER_SCENE_PER_ITERATION = 250
MIN_ACCEPTED_SCORE_PER_SCENE = 0.72

# Optional: keep these scene folder names out of training and use them for
# cross-scene evaluation if labels exist.
HELD_OUT_SCENES: Set[str] = set()

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


# =============================================================================
# C. Filesystem and dataset utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
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


def normalize_folder_name(folder_name: str) -> str:
    normalized = folder_name.strip().lower().replace("-", "_").replace(" ", "_")
    return FOLDER_CLASS_ALIASES.get(normalized, normalized)


def infer_scene(image_path: Path, image_root: Path) -> str:
    try:
        relative = image_path.resolve().relative_to(image_root.resolve())
    except ValueError:
        return "root_scene"
    if len(relative.parts) <= 1:
        return "root_scene"
    return relative.parts[0]


def expected_class_from_folder(image_path: Path, image_root: Path) -> Optional[int]:
    """
    Return class id from the image's direct parent folder.

    Images directly inside NEW_IMAGES return None. Unknown class folders return
    -1 so they can be rejected or penalized explicitly.
    """
    try:
        relative = image_path.resolve().relative_to(image_root.resolve())
    except ValueError:
        relative = image_path

    if len(relative.parts) < 2:
        return None

    class_name = normalize_folder_name(relative.parts[-2])
    return CLASS_NAME_TO_ID.get(class_name, -1)


def find_label_for_image(image_path: Path, image_root: Path) -> Optional[Path]:
    """
    Find a YOLO label file for an image.

    Supports common layouts:
    - same folder as image
    - sibling labels folder next to images folder
    - NEW_IMAGES/<scene>/labels matching NEW_IMAGES/<scene>/images
    """
    same_folder = image_path.with_suffix(".txt")
    if same_folder.exists():
        return same_folder

    candidates = []
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            label_parts = parts[:]
            label_parts[index] = "labels"
            candidates.append(Path(*label_parts).with_suffix(".txt"))

    try:
        relative = image_path.resolve().relative_to(image_root.resolve())
        candidates.append(image_root / "labels" / relative.with_suffix(".txt").name)
        if len(relative.parts) > 1:
            candidates.append(image_root / relative.parts[0] / "labels" / relative.with_suffix(".txt").name)
    except ValueError:
        pass

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def count_label_objects(label_file: Path) -> Tuple[int, int, Dict[int, int]]:
    valid_rows = 0
    bad_rows = 0
    class_counts = {class_id: 0 for class_id in CLASS_NAMES}

    if not label_file.exists() or label_file.stat().st_size == 0:
        return 0, 0, class_counts

    for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            bad_rows += 1
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            bad_rows += 1
            continue
        if class_id not in CLASS_NAMES:
            bad_rows += 1
            continue
        if not all(0.0 <= value <= 1.0 for value in values):
            bad_rows += 1
            continue
        if values[2] <= 0.0 or values[3] <= 0.0:
            bad_rows += 1
            continue
        valid_rows += 1
        class_counts[class_id] += 1

    return valid_rows, bad_rows, class_counts


def inspect_yolo_dataset(dataset_root: Path) -> Dict[str, Dict[str, int]]:
    print(f"\nDataset inspection: {dataset_root}")
    summary: Dict[str, Dict[str, int]] = {}

    for split in ("train", "valid", "test"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        images = list_images(images_dir)
        labels = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []

        valid_objects = 0
        bad_rows = 0
        empty_labels = 0
        class_counts = {class_id: 0 for class_id in CLASS_NAMES}
        label_stems = {label.stem for label in labels}
        missing_labels = sum(1 for image in images if image.stem not in label_stems)

        for label_file in labels:
            object_count, bad_count, label_class_counts = count_label_objects(label_file)
            valid_objects += object_count
            bad_rows += bad_count
            if object_count == 0:
                empty_labels += 1
            for class_id, count in label_class_counts.items():
                class_counts[class_id] += count

        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": valid_objects,
            "empty_labels": empty_labels,
            "missing_labels": missing_labels,
            "bad_rows": bad_rows,
        }

        distribution = ", ".join(
            f"{CLASS_NAMES[class_id]}={class_counts[class_id]}"
            for class_id in sorted(CLASS_NAMES)
        )
        print(
            f"  {split}: images={len(images)}, labels={len(labels)}, "
            f"objects={valid_objects}, empty_labels={empty_labels}, "
            f"missing_labels={missing_labels}, bad_rows={bad_rows}"
        )
        print(f"    objects_by_class: {distribution}")

    return summary


def safe_write_data_yaml(data_yaml_path: Path, dataset_root: Path) -> None:
    ensure_dir(data_yaml_path.parent)
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for class_id in sorted(CLASS_NAMES):
        lines.append(f"  {class_id}: {CLASS_NAMES[class_id]}")
    data_yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote data.yaml: {data_yaml_path}")


def unique_destination(base_dir: Path, source_path: Path, used_names: Set[str], prefix: str = "") -> Path:
    suffix = source_path.suffix
    stem = f"{prefix}{source_path.stem}"
    candidate_name = f"{stem}{suffix}"
    counter = 1

    while candidate_name.lower() in used_names or (base_dir / candidate_name).exists():
        candidate_name = f"{stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate_name.lower())
    return base_dir / candidate_name


# =============================================================================
# D. Box geometry and augmentation consistency
# =============================================================================

def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def xywh_to_xyxy(box: BoxPrediction) -> Tuple[float, float, float, float]:
    x1 = box.x_center - box.width / 2.0
    y1 = box.y_center - box.height / 2.0
    x2 = box.x_center + box.width / 2.0
    y2 = box.y_center + box.height / 2.0
    return x1, y1, x2, y2


def xyxy_to_box(class_id: int, xyxy: Tuple[float, float, float, float], confidence: float) -> BoxPrediction:
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return BoxPrediction(
        class_id=class_id,
        x_center=clamp01(x1 + width / 2.0),
        y_center=clamp01(y1 + height / 2.0),
        width=width,
        height=height,
        confidence=float(confidence),
    )


def box_iou(a: BoxPrediction, b: BoxPrediction) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def apply_augmentation(image: Image.Image, augmentation: str) -> Image.Image:
    if augmentation == "brightness_up":
        return ImageEnhance.Brightness(image).enhance(1.25)
    if augmentation == "brightness_down":
        return ImageEnhance.Brightness(image).enhance(0.75)
    if augmentation == "contrast_up":
        return ImageEnhance.Contrast(image).enhance(1.30)
    if augmentation == "horizontal_flip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if augmentation == "rotate90":
        return image.transpose(Image.Transpose.ROTATE_90)
    raise ValueError(f"Unknown augmentation: {augmentation}")


def invert_augmented_box(box: BoxPrediction, augmentation: str) -> BoxPrediction:
    """
    Map a prediction made on an augmented image back to original normalized space.

    Brightness/contrast do not change geometry. Horizontal flip and 90-degree
    rotation have exact normalized inverse transforms.
    """
    if augmentation in {"brightness_up", "brightness_down", "contrast_up"}:
        return box

    if augmentation == "horizontal_flip":
        return BoxPrediction(
            class_id=box.class_id,
            x_center=1.0 - box.x_center,
            y_center=box.y_center,
            width=box.width,
            height=box.height,
            confidence=box.confidence,
        )

    if augmentation == "rotate90":
        # PIL ROTATE_90 rotates counter-clockwise. Inverse to original:
        # original x = augmented y, original y = 1 - augmented x.
        return BoxPrediction(
            class_id=box.class_id,
            x_center=box.y_center,
            y_center=1.0 - box.x_center,
            width=box.height,
            height=box.width,
            confidence=box.confidence,
        )

    raise ValueError(f"Unknown augmentation: {augmentation}")


def result_to_boxes(result) -> List[BoxPrediction]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xywhn = boxes.xywhn.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    predictions: List[BoxPrediction] = []

    for box, class_value, confidence in zip(xywhn, classes, confidences):
        x_center, y_center, width, height = box
        predictions.append(
            BoxPrediction(
                class_id=int(class_value),
                x_center=clamp01(float(x_center)),
                y_center=clamp01(float(y_center)),
                width=clamp01(float(width)),
                height=clamp01(float(height)),
                confidence=float(confidence),
            )
        )
    return predictions


def predict_image(model: YOLO, image_path: Path, conf: float) -> List[BoxPrediction]:
    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=IMGSZ,
        device=DEVICE,
        save=False,
        verbose=False,
    )
    if not results:
        return []
    return result_to_boxes(results[0])


def predict_augmented_image(
    model: YOLO,
    image_path: Path,
    augmentation: str,
    conf: float,
    temp_dir: Path,
) -> List[BoxPrediction]:
    image = Image.open(image_path).convert("RGB")
    augmented = apply_augmentation(image, augmentation)
    augmented_path = temp_dir / f"{image_path.stem}_{augmentation}.jpg"
    augmented.save(augmented_path, quality=95)

    augmented_boxes = predict_image(model, augmented_path, conf=conf)
    return [invert_augmented_box(box, augmentation) for box in augmented_boxes]


def compute_consistency_support(
    original_box: BoxPrediction,
    augmented_predictions: Sequence[List[BoxPrediction]],
) -> int:
    support = 1
    for boxes in augmented_predictions:
        matched = any(
            candidate.class_id == original_box.class_id
            and box_iou(original_box, candidate) >= CONSISTENCY_IOU_THRESHOLD
            for candidate in boxes
        )
        if matched:
            support += 1
    return support


# =============================================================================
# E. Dynamic thresholds and composite quality scoring
# =============================================================================

def derive_class_thresholds(model: YOLO, data_yaml: Path) -> Dict[int, float]:
    """
    Build adaptive class confidence thresholds from validation AP.

    Classes with lower validation AP receive stricter thresholds. If per-class
    validation metrics are unavailable, all classes use BASE_CONF_THRESHOLD.
    """
    thresholds = {class_id: BASE_CONF_THRESHOLD for class_id in CLASS_NAMES}
    if not data_yaml.exists():
        print(f"Validation data.yaml not found, using default thresholds: {thresholds}")
        return thresholds

    try:
        metrics = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=IMGSZ,
            device=DEVICE,
            workers=WORKERS,
            verbose=False,
        )
        maps = getattr(metrics.box, "maps", None)
        if maps is None:
            print(f"Per-class AP unavailable, using default thresholds: {thresholds}")
            return thresholds

        for class_id in CLASS_NAMES:
            class_ap = float(maps[class_id]) if class_id < len(maps) else TARGET_CLASS_MAP
            adjustment = max(0.0, TARGET_CLASS_MAP - class_ap) * THRESHOLD_ADJUSTMENT_STRENGTH
            thresholds[class_id] = min(
                MAX_CONF_THRESHOLD,
                max(MIN_CONF_THRESHOLD, BASE_CONF_THRESHOLD + adjustment),
            )
    except Exception as exc:
        print(f"WARNING: Could not derive class thresholds from validation metrics: {exc}")
        print(f"Using default thresholds: {thresholds}")

    readable = ", ".join(
        f"{CLASS_NAMES[class_id]}={thresholds[class_id]:.3f}"
        for class_id in sorted(thresholds)
    )
    print(f"\nDynamic class-wise thresholds: {readable}")
    return thresholds


def edge_contact_score(box: BoxPrediction) -> Tuple[float, bool]:
    x1, y1, x2, y2 = xywh_to_xyxy(box)
    touches_edge = (
        x1 < EDGE_MARGIN
        or y1 < EDGE_MARGIN
        or x2 > 1.0 - EDGE_MARGIN
        or y2 > 1.0 - EDGE_MARGIN
    )
    return (0.0 if touches_edge else 1.0), touches_edge


def geometry_score(box: BoxPrediction) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    area = box.width * box.height
    if box.width <= 0.0 or box.height <= 0.0:
        return 0.0, ["non_positive_box"]

    if area < MIN_BOX_AREA:
        reasons.append("box_area_too_small")
        area_score = 0.0
    elif area < IDEAL_MIN_BOX_AREA:
        area_score = area / IDEAL_MIN_BOX_AREA
    elif area <= IDEAL_MAX_BOX_AREA:
        area_score = 1.0
    elif area <= MAX_BOX_AREA:
        area_score = max(0.0, 1.0 - (area - IDEAL_MAX_BOX_AREA) / (MAX_BOX_AREA - IDEAL_MAX_BOX_AREA))
    else:
        reasons.append("box_area_too_large")
        area_score = 0.0

    aspect_ratio = max(box.width / box.height, box.height / box.width)
    if aspect_ratio > MAX_ASPECT_RATIO:
        reasons.append("bad_aspect_ratio")
        aspect_score = 0.0
    else:
        aspect_score = max(0.0, 1.0 - (aspect_ratio - 1.0) / (MAX_ASPECT_RATIO - 1.0))

    return 0.65 * area_score + 0.35 * aspect_score, reasons


def class_folder_score(box: BoxPrediction, expected_class_id: Optional[int]) -> Tuple[float, List[str]]:
    if expected_class_id is None:
        return 0.75, []
    if expected_class_id == -1:
        return 0.0, ["unknown_folder_class"]
    if box.class_id != expected_class_id:
        return 0.0, ["folder_class_mismatch"]
    return 1.0, []


def compute_quality_score(
    box: BoxPrediction,
    expected_class_id: Optional[int],
    class_thresholds: Dict[int, float],
) -> BoxPrediction:
    reasons: List[str] = []
    threshold = class_thresholds.get(box.class_id, BASE_CONF_THRESHOLD)
    confidence_score = min(1.0, box.confidence / max(threshold, 1e-6))

    geom_score, geom_reasons = geometry_score(box)
    edge_score, touches_edge = edge_contact_score(box)
    folder_score, folder_reasons = class_folder_score(box, expected_class_id)

    if box.class_id not in CLASS_NAMES:
        reasons.append("unknown_class")
    if box.confidence < threshold:
        reasons.append("below_dynamic_class_threshold")
    if touches_edge:
        reasons.append("touches_image_edge")
    reasons.extend(geom_reasons)
    reasons.extend(folder_reasons)

    quality_score = (
        0.42 * confidence_score
        + 0.25 * geom_score
        + 0.15 * edge_score
        + 0.18 * folder_score
    )

    box.quality_score = quality_score
    box.reasons = reasons
    return box


def accept_box(box: BoxPrediction) -> bool:
    if box.class_id not in CLASS_NAMES:
        return False
    if box.support < MIN_AUGMENTATION_SUPPORT:
        box.reasons.append("low_consistency_support")
        return False
    if box.quality_score < QUALITY_THRESHOLD:
        box.reasons.append("low_quality_score")
        return False
    if REJECT_IF_FOLDER_CLASS_MISMATCH and "folder_class_mismatch" in box.reasons:
        return False
    return True


# =============================================================================
# F. Pseudo-label generation with consistency and quality filtering
# =============================================================================

def generate_consistent_pseudo_labels(
    model_path: Path,
    image_folder: Path,
    output_root: Path,
    class_thresholds: Dict[int, float],
    iteration: int,
    skip_scenes: Set[str],
) -> List[ImagePseudoResult]:
    pseudo_images_dir = output_root / f"iteration_{iteration}" / "pseudo_labeled_images"
    pseudo_labels_dir = output_root / f"iteration_{iteration}" / "pseudo_labels_filtered"
    rejected_images_dir = output_root / f"iteration_{iteration}" / "pseudo_rejected_images"
    rejected_labels_dir = output_root / f"iteration_{iteration}" / "pseudo_rejected_labels"
    report_path = output_root / f"iteration_{iteration}" / "pseudo_label_quality_report.csv"

    clear_dir(pseudo_images_dir)
    clear_dir(pseudo_labels_dir)
    clear_dir(rejected_images_dir)
    clear_dir(rejected_labels_dir)
    ensure_dir(report_path.parent)

    images = [image for image in list_images(image_folder) if infer_scene(image, image_folder) not in skip_scenes]
    print(f"\nIteration {iteration}: pseudo-labeling {len(images)} images from {image_folder}")
    if not images:
        return []

    model = YOLO(str(model_path))
    results: List[ImagePseudoResult] = []
    used_accepted_names: Set[str] = set()
    used_rejected_names: Set[str] = set()

    with tempfile.TemporaryDirectory(prefix="yolo_aug_") as temp_name:
        temp_dir = Path(temp_name)
        for index, image_path in enumerate(images, start=1):
            scene = infer_scene(image_path, image_folder)
            expected_class_id = expected_class_from_folder(image_path, image_folder)
            min_conf = min(class_thresholds.values())

            original_boxes = predict_image(model, image_path, conf=min_conf)
            if not original_boxes:
                results.append(
                    ImagePseudoResult(
                        image_path=image_path,
                        scene=scene,
                        accepted_boxes=[],
                        rejected_boxes=[],
                        status="rejected",
                        reasons=["no_detections"],
                    )
                )
                continue

            augmented_predictions = [
                predict_augmented_image(model, image_path, augmentation, conf=min_conf, temp_dir=temp_dir)
                for augmentation in CONSISTENCY_AUGMENTATIONS
            ]

            accepted_boxes: List[BoxPrediction] = []
            rejected_boxes: List[BoxPrediction] = []

            for box in original_boxes:
                box.support = compute_consistency_support(box, augmented_predictions)
                scored_box = compute_quality_score(box, expected_class_id, class_thresholds)
                if accept_box(scored_box):
                    accepted_boxes.append(scored_box)
                else:
                    rejected_boxes.append(scored_box)

            if len(accepted_boxes) > MAX_BOXES_PER_IMAGE:
                for box in accepted_boxes:
                    box.reasons.append("too_many_boxes")
                rejected_boxes.extend(accepted_boxes)
                accepted_boxes = []

            if accepted_boxes:
                destination_image = unique_destination(pseudo_images_dir, image_path, used_accepted_names)
                destination_label = pseudo_labels_dir / f"{destination_image.stem}.txt"
                copy_file(image_path, destination_image)
                write_yolo_label(destination_label, accepted_boxes, include_score=False)
                status = "accepted"
                reasons: List[str] = []
            else:
                destination_image = unique_destination(rejected_images_dir, image_path, used_rejected_names)
                destination_label = rejected_labels_dir / f"{destination_image.stem}.txt"
                copy_file(image_path, destination_image)
                write_yolo_label(destination_label, rejected_boxes, include_score=True)
                status = "rejected"
                reasons = sorted({reason for box in rejected_boxes for reason in box.reasons}) or ["no_valid_box"]

            results.append(
                ImagePseudoResult(
                    image_path=image_path,
                    scene=scene,
                    accepted_boxes=accepted_boxes,
                    rejected_boxes=rejected_boxes,
                    status=status,
                    reasons=reasons,
                )
            )

            if index % 50 == 0:
                print(f"  processed {index}/{len(images)} images")

    write_quality_report(report_path, results)
    print_pseudo_summary(results, title=f"Iteration {iteration} pseudo-label summary")
    print(f"Filtered pseudo-labels: {pseudo_labels_dir}")
    print(f"Quality report: {report_path}")
    return results


def write_yolo_label(label_path: Path, boxes: Sequence[BoxPrediction], include_score: bool) -> None:
    ensure_dir(label_path.parent)
    lines = []
    for box in boxes:
        base = (
            f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} "
            f"{box.width:.6f} {box.height:.6f}"
        )
        if include_score:
            base = f"{base} conf={box.confidence:.6f} score={box.quality_score:.6f} support={box.support}"
            if box.reasons:
                base = f"{base} reasons={'|'.join(sorted(set(box.reasons)))}"
        lines.append(base)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def write_quality_report(report_path: Path, results: Sequence[ImagePseudoResult]) -> None:
    ensure_dir(report_path.parent)
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "scene",
                "status",
                "class_id",
                "class_name",
                "confidence",
                "quality_score",
                "support",
                "x_center",
                "y_center",
                "width",
                "height",
                "reasons",
            ],
        )
        writer.writeheader()
        for result in results:
            boxes = result.accepted_boxes if result.accepted_boxes else result.rejected_boxes
            if not boxes:
                writer.writerow(
                    {
                        "image": str(result.image_path),
                        "scene": result.scene,
                        "status": result.status,
                        "class_id": "",
                        "class_name": "",
                        "confidence": "",
                        "quality_score": "",
                        "support": "",
                        "x_center": "",
                        "y_center": "",
                        "width": "",
                        "height": "",
                        "reasons": "|".join(result.reasons),
                    }
                )
                continue
            for box in boxes:
                writer.writerow(
                    {
                        "image": str(result.image_path),
                        "scene": result.scene,
                        "status": result.status,
                        "class_id": box.class_id,
                        "class_name": CLASS_NAMES.get(box.class_id, "unknown"),
                        "confidence": f"{box.confidence:.6f}",
                        "quality_score": f"{box.quality_score:.6f}",
                        "support": box.support,
                        "x_center": f"{box.x_center:.6f}",
                        "y_center": f"{box.y_center:.6f}",
                        "width": f"{box.width:.6f}",
                        "height": f"{box.height:.6f}",
                        "reasons": "|".join(sorted(set(box.reasons))),
                    }
                )


def print_pseudo_summary(results: Sequence[ImagePseudoResult], title: str) -> None:
    accepted_images = [result for result in results if result.status == "accepted"]
    rejected_images = [result for result in results if result.status != "accepted"]
    accepted_by_class = {class_id: 0 for class_id in CLASS_NAMES}
    rejected_by_class = {class_id: 0 for class_id in CLASS_NAMES}
    accepted_by_scene: Dict[str, int] = {}
    rejected_by_scene: Dict[str, int] = {}

    for result in accepted_images:
        accepted_by_scene[result.scene] = accepted_by_scene.get(result.scene, 0) + 1
        for box in result.accepted_boxes:
            if box.class_id in accepted_by_class:
                accepted_by_class[box.class_id] += 1

    for result in rejected_images:
        rejected_by_scene[result.scene] = rejected_by_scene.get(result.scene, 0) + 1
        for box in result.rejected_boxes:
            if box.class_id in rejected_by_class:
                rejected_by_class[box.class_id] += 1

    print(f"\n{title}:")
    print(f"  accepted_images={len(accepted_images)}, rejected_images={len(rejected_images)}")
    print(
        "  accepted_boxes_by_class: "
        + ", ".join(f"{CLASS_NAMES[c]}={accepted_by_class[c]}" for c in sorted(CLASS_NAMES))
    )
    print(
        "  rejected_boxes_by_class: "
        + ", ".join(f"{CLASS_NAMES[c]}={rejected_by_class[c]}" for c in sorted(CLASS_NAMES))
    )
    print(f"  accepted_images_by_scene: {dict(sorted(accepted_by_scene.items()))}")
    print(f"  rejected_images_by_scene: {dict(sorted(rejected_by_scene.items()))}")


# =============================================================================
# G. Cross-scene selection and combined dataset construction
# =============================================================================

def select_pseudo_images_for_iteration(
    results: Sequence[ImagePseudoResult],
    output_root: Path,
    iteration: int,
) -> Tuple[Path, Path]:
    """
    Select high-quality pseudo-labeled images with per-scene caps.

    This keeps one scene from dominating self-training and makes the added data
    more useful for cross-scene generalization.
    """
    selected_images_dir = output_root / f"iteration_{iteration}" / "selected_pseudo_images"
    selected_labels_dir = output_root / f"iteration_{iteration}" / "selected_pseudo_labels"
    clear_dir(selected_images_dir)
    clear_dir(selected_labels_dir)

    accepted = [result for result in results if result.status == "accepted" and result.accepted_boxes]
    accepted.sort(
        key=lambda result: (
            result.scene,
            -sum(box.quality_score for box in result.accepted_boxes) / len(result.accepted_boxes),
        )
    )

    per_scene_counts: Dict[str, int] = {}
    used_names: Set[str] = set()
    source_pseudo_images = output_root / f"iteration_{iteration}" / "pseudo_labeled_images"
    source_pseudo_labels = output_root / f"iteration_{iteration}" / "pseudo_labels_filtered"

    selected = 0
    for result in accepted:
        mean_score = sum(box.quality_score for box in result.accepted_boxes) / len(result.accepted_boxes)
        if mean_score < MIN_ACCEPTED_SCORE_PER_SCENE:
            continue
        scene_count = per_scene_counts.get(result.scene, 0)
        if scene_count >= MAX_ACCEPTED_IMAGES_PER_SCENE_PER_ITERATION:
            continue

        source_image = source_pseudo_images / result.image_path.name
        source_label = source_pseudo_labels / f"{result.image_path.stem}.txt"
        if not source_image.exists() or not source_label.exists():
            # The copied pseudo image may have a unique suffix, so find by label content fallback.
            matches = list(source_pseudo_images.glob(f"{result.image_path.stem}*{result.image_path.suffix}"))
            if not matches:
                continue
            source_image = matches[0]
            source_label = source_pseudo_labels / f"{source_image.stem}.txt"
            if not source_label.exists():
                continue

        destination_image = unique_destination(selected_images_dir, source_image, used_names, prefix=f"{result.scene}_")
        destination_label = selected_labels_dir / f"{destination_image.stem}.txt"
        copy_file(source_image, destination_image)
        copy_file(source_label, destination_label)
        per_scene_counts[result.scene] = scene_count + 1
        selected += 1

    print(f"\nSelected pseudo images for iteration {iteration}: {selected}")
    print(f"Selected by scene: {dict(sorted(per_scene_counts.items()))}")
    return selected_images_dir, selected_labels_dir


def copy_labeled_split(
    source_images: Path,
    source_labels: Path,
    destination_images: Path,
    destination_labels: Path,
    prefix: str = "",
) -> int:
    ensure_dir(destination_images)
    ensure_dir(destination_labels)
    used_names = {path.name.lower() for path in destination_images.iterdir() if path.is_file()}
    copied = 0

    for image_path in list_images(source_images):
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        object_count, _, _ = count_label_objects(label_path)
        if object_count == 0:
            continue
        destination_image = unique_destination(destination_images, image_path, used_names, prefix=prefix)
        destination_label = destination_labels / f"{destination_image.stem}.txt"
        copy_file(image_path, destination_image)
        copy_file(label_path, destination_label)
        copied += 1

    return copied


def split_pseudo_images(
    pseudo_images: Path,
    pseudo_labels: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Path]]:
    valid_images = []
    for image_path in list_images(pseudo_images):
        label_path = pseudo_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        object_count, _, _ = count_label_objects(label_path)
        if object_count > 0:
            valid_images.append(image_path)

    rng = random.Random(seed)
    rng.shuffle(valid_images)
    train_cut = int(len(valid_images) * train_ratio)
    val_cut = int(len(valid_images) * (train_ratio + val_ratio))
    return {
        "train": valid_images[:train_cut],
        "valid": valid_images[train_cut:val_cut],
        "test": valid_images[val_cut:],
    }


def copy_selected_pseudo_split(
    image_paths: Sequence[Path],
    source_labels: Path,
    destination_images: Path,
    destination_labels: Path,
    prefix: str,
) -> int:
    ensure_dir(destination_images)
    ensure_dir(destination_labels)
    used_names = {path.name.lower() for path in destination_images.iterdir() if path.is_file()}
    copied = 0

    for image_path in image_paths:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        object_count, _, _ = count_label_objects(label_path)
        if object_count == 0:
            continue
        destination_image = unique_destination(destination_images, image_path, used_names, prefix=prefix)
        destination_label = destination_labels / f"{destination_image.stem}.txt"
        copy_file(image_path, destination_image)
        copy_file(label_path, destination_label)
        copied += 1

    return copied


def build_combined_dataset(
    original_dataset: Path,
    pseudo_sources: Sequence[Tuple[Path, Path]],
    combined_root: Path,
) -> Path:
    clear_dir(combined_root)

    for split in ("train", "valid", "test"):
        ensure_dir(combined_root / split / "images")
        ensure_dir(combined_root / split / "labels")

    copied_original = {
        "train": copy_labeled_split(
            original_dataset / "train" / "images",
            original_dataset / "train" / "labels",
            combined_root / "train" / "images",
            combined_root / "train" / "labels",
        ),
        "valid": copy_labeled_split(
            original_dataset / "valid" / "images",
            original_dataset / "valid" / "labels",
            combined_root / "valid" / "images",
            combined_root / "valid" / "labels",
        ),
        "test": copy_labeled_split(
            original_dataset / "test" / "images",
            original_dataset / "test" / "labels",
            combined_root / "test" / "images",
            combined_root / "test" / "labels",
        ),
    }

    copied_pseudo = {"train": 0, "valid": 0, "test": 0}
    for source_index, (pseudo_images, pseudo_labels) in enumerate(pseudo_sources, start=1):
        pseudo_splits = split_pseudo_images(
            pseudo_images=pseudo_images,
            pseudo_labels=pseudo_labels,
            train_ratio=PSEUDO_TRAIN_RATIO,
            val_ratio=PSEUDO_VAL_RATIO,
            seed=SEED + source_index,
        )
        for split, image_paths in pseudo_splits.items():
            copied_pseudo[split] += copy_selected_pseudo_split(
                image_paths=image_paths,
                source_labels=pseudo_labels,
                destination_images=combined_root / split / "images",
                destination_labels=combined_root / split / "labels",
                prefix=f"pseudo{source_index}_",
            )

    data_yaml = combined_root / "data.yaml"
    safe_write_data_yaml(data_yaml, combined_root)

    print("\nCombined dataset copied:")
    for split in ("train", "valid", "test"):
        print(
            f"  {split}: original={copied_original[split]}, "
            f"pseudo={copied_pseudo[split]}, total={copied_original[split] + copied_pseudo[split]}"
        )

    inspect_yolo_dataset(combined_root)
    return data_yaml


# =============================================================================
# H. Training, demo prediction, and cross-scene evaluation
# =============================================================================

def train_yolov8n(data_yaml: Path, output_project: Path, run_name: str) -> YOLO:
    print(f"\nRetraining YOLOv8n with data: {data_yaml}")
    model = YOLO(YOLO_BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(output_project),
        name=run_name,
        exist_ok=True,
        seed=SEED,
        amp=False,
    )
    return model


def run_demo_predictions(model: YOLO, valid_images_dir: Path, output_project: Path, run_name: str) -> None:
    if not list_images(valid_images_dir):
        print(f"Skipping demo predictions because no validation images exist: {valid_images_dir}")
        return

    print(f"\nSaving demo predictions from: {valid_images_dir}")
    model.predict(
        source=str(valid_images_dir),
        conf=0.25,
        imgsz=IMGSZ,
        device=DEVICE,
        save=True,
        project=str(output_project),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )


def build_scene_eval_dataset(scene: str, images: Sequence[Path], image_root: Path, output_root: Path) -> Optional[Path]:
    scene_root = output_root / "scene_eval" / scene
    clear_dir(scene_root)
    ensure_dir(scene_root / "train" / "images")
    ensure_dir(scene_root / "train" / "labels")
    ensure_dir(scene_root / "valid" / "images")
    ensure_dir(scene_root / "valid" / "labels")
    images_dir = scene_root / "test" / "images"
    labels_dir = scene_root / "test" / "labels"
    ensure_dir(images_dir)
    ensure_dir(labels_dir)

    used_names: Set[str] = set()
    copied = 0
    for image_path in images:
        label_path = find_label_for_image(image_path, image_root)
        if label_path is None:
            continue
        object_count, _, _ = count_label_objects(label_path)
        if object_count == 0:
            continue
        destination_image = unique_destination(images_dir, image_path, used_names)
        destination_label = labels_dir / f"{destination_image.stem}.txt"
        copy_file(image_path, destination_image)
        copy_file(label_path, destination_label)
        copied += 1

    if copied == 0:
        return None

    data_yaml = scene_root / "data.yaml"
    safe_write_data_yaml(data_yaml, scene_root)
    return data_yaml


def evaluate_cross_scene(model: YOLO, image_root: Path, output_root: Path) -> None:
    images_by_scene: Dict[str, List[Path]] = {}
    for image_path in list_images(image_root):
        scene = infer_scene(image_path, image_root)
        images_by_scene.setdefault(scene, []).append(image_path)

    if not images_by_scene:
        print("\nNo scenes found for cross-scene evaluation.")
        return

    print("\nCross-scene evaluation:")
    for scene, images in sorted(images_by_scene.items()):
        scene_data_yaml = build_scene_eval_dataset(scene, images, image_root, output_root)
        if scene_data_yaml is None:
            # Unlabeled scene: report prediction volume for debugging.
            predictions = 0
            for image_path in images[: min(50, len(images))]:
                predictions += len(predict_image(model, image_path, conf=0.25))
            print(
                f"  {scene}: no labels found, skipped mAP. "
                f"sampled_predictions={predictions} on {min(50, len(images))}/{len(images)} images"
            )
            continue

        metrics = model.val(
            data=str(scene_data_yaml),
            split="test",
            imgsz=IMGSZ,
            device=DEVICE,
            workers=WORKERS,
            verbose=False,
        )
        print(
            f"  {scene}: images={len(images)}, "
            f"P={metrics.box.mp:.4f}, R={metrics.box.mr:.4f}, "
            f"mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}"
        )


def evaluate_held_out_scenes(model: YOLO, image_root: Path, output_root: Path) -> None:
    if not HELD_OUT_SCENES:
        return

    print("\nHeld-out scene evaluation:")
    for scene in sorted(HELD_OUT_SCENES):
        images = [image for image in list_images(image_root) if infer_scene(image, image_root) == scene]
        if not images:
            print(f"  {scene}: no images found")
            continue
        scene_data_yaml = build_scene_eval_dataset(f"heldout_{scene}", images, image_root, output_root)
        if scene_data_yaml is None:
            print(f"  {scene}: labels not found, cannot compute mAP")
            continue
        metrics = model.val(
            data=str(scene_data_yaml),
            split="test",
            imgsz=IMGSZ,
            device=DEVICE,
            workers=WORKERS,
            verbose=False,
        )
        print(
            f"  {scene}: P={metrics.box.mp:.4f}, R={metrics.box.mr:.4f}, "
            f"mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}"
        )


# =============================================================================
# I. Main orchestration
# =============================================================================

def main() -> None:
    print("Consistency self-training configuration:")
    print(f"  TRAINED_MODEL:    {TRAINED_MODEL}")
    print(f"  ORIGINAL_DATASET: {ORIGINAL_DATASET}")
    print(f"  NEW_IMAGES:       {NEW_IMAGES}")
    print(f"  OUTPUT_ROOT:      {OUTPUT_ROOT}")
    print(f"  EPOCHS:           {EPOCHS}")
    print(f"  IMGSZ:            {IMGSZ}")
    print(f"  BATCH:            {BATCH}")
    print(f"  DEVICE:           {DEVICE}")
    print(f"  HELD_OUT_SCENES:  {sorted(HELD_OUT_SCENES)}")

    if not TRAINED_MODEL.exists():
        raise FileNotFoundError(f"Trained model not found: {TRAINED_MODEL}")
    if not ORIGINAL_DATASET.exists():
        raise FileNotFoundError(f"Original dataset not found: {ORIGINAL_DATASET}")
    if not NEW_IMAGES.exists():
        raise FileNotFoundError(f"New image folder not found: {NEW_IMAGES}")

    ensure_dir(OUTPUT_ROOT)
    inspect_yolo_dataset(ORIGINAL_DATASET)

    original_data_yaml = ORIGINAL_DATASET / "data.yaml"
    baseline_model = YOLO(str(TRAINED_MODEL))
    class_thresholds = derive_class_thresholds(baseline_model, original_data_yaml)

    pseudo_sources: List[Tuple[Path, Path]] = []
    current_model_path = TRAINED_MODEL

    for iteration in range(1, SELF_TRAINING_ITERATIONS + 1):
        results = generate_consistent_pseudo_labels(
            model_path=current_model_path,
            image_folder=NEW_IMAGES,
            output_root=OUTPUT_ROOT,
            class_thresholds=class_thresholds,
            iteration=iteration,
            skip_scenes=HELD_OUT_SCENES,
        )
        selected_images, selected_labels = select_pseudo_images_for_iteration(
            results=results,
            output_root=OUTPUT_ROOT,
            iteration=iteration,
        )
        pseudo_sources.append((selected_images, selected_labels))

        combined_root = OUTPUT_ROOT / f"combined_dataset_iter_{iteration}"
        combined_data_yaml = build_combined_dataset(
            original_dataset=ORIGINAL_DATASET,
            pseudo_sources=pseudo_sources,
            combined_root=combined_root,
        )

        run_project = OUTPUT_ROOT / "runs"
        run_name = f"consistency_retrain_iter_{iteration}"
        trained_model = train_yolov8n(combined_data_yaml, run_project, run_name)

        run_demo_predictions(
            model=trained_model,
            valid_images_dir=combined_root / "valid" / "images",
            output_project=run_project,
            run_name=f"demo_predictions_iter_{iteration}",
        )

        current_model_path = run_project / run_name / "weights" / "best.pt"
        if not current_model_path.exists():
            raise FileNotFoundError(f"Retrained best.pt not found: {current_model_path}")

        # Refresh thresholds after each iteration so later pseudo-labeling adapts
        # to the model's new class-wise validation performance.
        class_thresholds = derive_class_thresholds(YOLO(str(current_model_path)), combined_data_yaml)

    final_model = YOLO(str(current_model_path))
    evaluate_cross_scene(final_model, NEW_IMAGES, OUTPUT_ROOT)
    evaluate_held_out_scenes(final_model, NEW_IMAGES, OUTPUT_ROOT)

    print("\nPipeline complete.")
    print(f"Final model: {current_model_path}")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

