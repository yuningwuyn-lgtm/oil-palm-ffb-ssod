"""
Strict zero-shot third-domain evaluation for the Mendeley ordinal FFB dataset.

The Mendeley dataset is an image-level classification benchmark. It is never
used for training, pseudo-labeling, threshold fitting, or hard-negative mining.
Only the authors' official Testing.txt split is reported as the paper result.

Dataset: An Ordinal Dataset for Ripeness Level Classification in Oil Palm
Fruit Quality Grading, DOI: 10.17632/424y96m6sw.1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image
from ultralytics import YOLO

from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    average_image_hash,
    hamming_distance,
    list_images,
    normalized_base_filename,
    predict_one,
)


MENDELEY_CLASSES = {
    0: "Immature",
    1: "PartiallyRipe",
    2: "FullyRipe",
    3: "OverRipe",
    4: "Decayed",
}
FOLDER_TO_CLASS = {
    "0Immature": 0,
    "1PartiallyRipe": 1,
    "2FullyRipe": 2,
    "3OverRipe": 3,
    "4Decayed": 4,
}
# Canonical detector labels: abnormal, empty, overripe, ripe, under_ripe, unripe.
# Empty bunch has no equivalent class in this dataset and is excluded when
# selecting the best mapped detection.
DETECTOR_TO_MENDELEY = {
    0: 4,
    2: 3,
    3: 2,
    4: 1,
    5: 0,
}


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def collect_official_test_records(dataset_root: Path) -> list[dict]:
    images_root = dataset_root / "Images"
    split_file = dataset_root / "Train_val_test_split" / "Testing.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Official test list not found: {split_file}")

    filenames = {
        line.strip()
        for line in split_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    indexed_images = {image.name.lower(): image for image in list_images(images_root)}
    records = []
    missing = []
    for filename in sorted(filenames):
        image = indexed_images.get(filename.lower())
        if image is None:
            missing.append(filename)
            continue
        folder_name = image.parent.name
        if folder_name not in FOLDER_TO_CLASS:
            raise ValueError(f"Unknown Mendeley folder label: {folder_name}")
        class_id = FOLDER_TO_CLASS[folder_name]
        records.append(
            {
                "image_path": image,
                "gt_class_id": class_id,
                "gt_class": MENDELEY_CLASSES[class_id],
            }
        )
    if missing:
        raise FileNotFoundError(f"{len(missing)} official test images are missing; first={missing[0]}")
    return records


def evaluate_zero_shot(
    model_path: Path,
    records: Sequence[dict],
    imgsz: int,
    device: str,
    min_prediction_conf: float,
) -> list[dict]:
    model = YOLO(str(model_path))
    rows = []
    for index, record in enumerate(records, start=1):
        predictions = predict_one(
            model,
            record["image_path"],
            imgsz=imgsz,
            device=device,
            conf=min_prediction_conf,
        )
        mapped_predictions = [box for box in predictions if box.class_id in DETECTOR_TO_MENDELEY]
        best = max(mapped_predictions, key=lambda box: box.confidence, default=None)
        raw_id = best.class_id if best is not None else ""
        mapped_id = DETECTOR_TO_MENDELEY.get(raw_id, "") if best is not None else ""
        rows.append(
            {
                "image": str(record["image_path"]),
                "official_split": "test",
                "gt_class_id": record["gt_class_id"],
                "gt_class": record["gt_class"],
                "raw_pred_class_id": raw_id,
                "raw_pred_class": CANONICAL_CLASSES.get(raw_id, "no_detection") if best is not None else "no_detection",
                "pred_class_id": mapped_id,
                "pred_class": MENDELEY_CLASSES.get(mapped_id, "no_detection"),
                "confidence": f"{best.confidence:.6f}" if best is not None else "0.000000",
            }
        )
        if index % 100 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)} images")
    return rows


def summarize_predictions(rows: Sequence[dict], threshold: float) -> dict:
    confusion = Counter()
    gt_counts = Counter()
    pred_counts = Counter()
    correct = 0
    detected = 0
    for row in rows:
        gt_id = int(row["gt_class_id"])
        pred_id = row["pred_class_id"]
        gt_counts[gt_id] += 1
        if pred_id == "" or float(row["confidence"]) < threshold:
            pred_id = "no_detection"
        else:
            pred_id = int(pred_id)
            detected += 1
            pred_counts[pred_id] += 1
            correct += int(pred_id == gt_id)
        confusion[(gt_id, pred_id)] += 1

    class_rows = []
    for class_id, class_name in MENDELEY_CLASSES.items():
        tp = confusion[(class_id, class_id)]
        fp = sum(confusion[(other, class_id)] for other in MENDELEY_CLASSES if other != class_id)
        fn = sum(count for (gt_id, pred_id), count in confusion.items() if gt_id == class_id and pred_id != class_id)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_images": gt_counts[class_id],
                "predicted_images": pred_counts[class_id],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": safe_div(2 * precision * recall, precision + recall),
            }
        )
    return {
        "summary": {
            "threshold": threshold,
            "images": len(rows),
            "detected_images": detected,
            "coverage": safe_div(detected, len(rows)),
            "top1_accuracy": safe_div(correct, len(rows)),
            "macro_precision": sum(row["precision"] for row in class_rows) / len(class_rows),
            "macro_recall": sum(row["recall"] for row in class_rows) / len(class_rows),
            "macro_f1": sum(row["f1"] for row in class_rows) / len(class_rows),
        },
        "classes": class_rows,
        "confusion": confusion,
    }


def write_confusion_matrix(path: Path, confusion: Counter) -> None:
    columns: list[int | str] = [*MENDELEY_CLASSES, "no_detection"]
    rows = []
    for gt_id, gt_name in MENDELEY_CLASSES.items():
        row = {"gt_class": gt_name}
        for pred_id in columns:
            pred_name = MENDELEY_CLASSES.get(pred_id, "no_detection")
            row[f"pred_{pred_name}"] = confusion[(gt_id, pred_id)]
        rows.append(row)
    write_csv(path, rows)


def duplicate_report(
    third_domain_images: Sequence[Path],
    reference_roots: Sequence[Path],
    output_csv: Path,
    hash_distance: int,
) -> dict:
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def difference_hash(path: Path, size: int = 16) -> int:
        with Image.open(path) as image:
            gray = image.convert("L").resize((size + 1, size))
            pixels = list(gray.getdata())
        result = 0
        for y_coord in range(size):
            for x_coord in range(size):
                offset = y_coord * (size + 1) + x_coord
                result = (result << 1) | int(pixels[offset] > pixels[offset + 1])
        return result

    third_by_name = defaultdict(list)
    third_by_hash = defaultdict(list)
    for image in third_domain_images:
        third_by_name[normalized_base_filename(image)].append(image)
        image_hash = average_image_hash(image)
        if image_hash is not None:
            third_by_hash[image_hash].append(image)

    rows = []
    reference_images = 0
    for root in reference_roots:
        if not root.exists():
            continue
        for reference in list_images(root):
            reference_images += 1
            for image in third_by_name.get(normalized_base_filename(reference), []):
                rows.append(
                    {
                        "match_type": "normalized_filename",
                        "distance": 0,
                        "dhash16_distance": (difference_hash(reference) ^ difference_hash(image)).bit_count(),
                        "exact_sha256": sha256(reference) == sha256(image),
                        "verified_duplicate": True,
                        "third_domain_image": str(image),
                        "reference_image": str(reference),
                        "reference_root": str(root),
                    }
                )
            reference_hash = average_image_hash(reference)
            if reference_hash is None:
                continue
            if hash_distance == 0:
                hash_candidates = [(reference_hash, third_by_hash.get(reference_hash, []))]
            else:
                hash_candidates = [
                    (image_hash, images)
                    for image_hash, images in third_by_hash.items()
                    if hamming_distance(reference_hash, image_hash) <= hash_distance
                ]
            for image_hash, images in hash_candidates:
                distance = hamming_distance(reference_hash, image_hash)
                for image in images:
                    dhash16_distance = (difference_hash(reference) ^ difference_hash(image)).bit_count()
                    exact_sha256 = sha256(reference) == sha256(image)
                    rows.append(
                        {
                            "match_type": "average_hash",
                            "distance": distance,
                            "dhash16_distance": dhash16_distance,
                            "exact_sha256": exact_sha256,
                            "verified_duplicate": exact_sha256 or dhash16_distance <= 10,
                            "third_domain_image": str(image),
                            "reference_image": str(reference),
                            "reference_root": str(root),
                        }
                    )

    unique_rows = list({tuple(row.items()): row for row in rows}.values())
    write_csv(
        output_csv,
        unique_rows,
        fieldnames=[
            "match_type",
            "distance",
            "dhash16_distance",
            "exact_sha256",
            "verified_duplicate",
            "third_domain_image",
            "reference_image",
            "reference_root",
        ],
    )
    return {
        "third_domain_images": len(third_domain_images),
        "reference_images": reference_images,
        "candidate_matches": len(unique_rows),
        "verified_duplicate_matches": sum(str(row["verified_duplicate"]).lower() == "true" for row in unique_rows),
        "hash_distance": hash_distance,
        "verification": "average-hash candidates confirmed by SHA256 or 16x16 dHash distance <= 10",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Mendeley ordinal FFB third-domain zero-shot evaluation.")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"LOCAL_DATA_ROOT\public_datasets\mendeley_ordinal_ffb\prepared\Dataset"))
    parser.add_argument("--model", type=Path, default=Path(r"LOCAL_DATA_ROOT\paper_stage2_formal\seed_42\model2_ssod_fp250_urpos640_yolov8n\runs\baseline_supervised\weights\best.pt"))
    parser.add_argument("--output-root", type=Path, default=Path(r"LOCAL_PROJECT_ROOT\paper_framework\reports\third_domain_mendeley_ordinal"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--min-prediction-conf", type=float, default=0.001)
    parser.add_argument("--thresholds", type=str, default="0.001,0.10,0.25,0.40,0.50,0.60,0.70")
    parser.add_argument("--duplicate-reference", type=Path, action="append", default=[])
    parser.add_argument("--duplicate-hash-distance", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = collect_official_test_records(args.dataset_root)
    if not records:
        raise ValueError(f"No official test records found: {args.dataset_root}")

    print(f"Mendeley third-domain root: {args.dataset_root}")
    print(f"Official split: test; images={len(records)}")
    print("This dataset is evaluation-only and will not be merged into any training source.")
    predictions = evaluate_zero_shot(
        model_path=args.model,
        records=records,
        imgsz=args.imgsz,
        device=args.device,
        min_prediction_conf=args.min_prediction_conf,
    )
    write_csv(args.output_root / "mendeley_ordinal_zero_shot_predictions.csv", predictions)

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    reports = {}
    threshold_rows = []
    for threshold in thresholds:
        report = summarize_predictions(predictions, threshold)
        reports[f"{threshold:.3f}"] = report
        threshold_rows.append(report["summary"])
    write_csv(args.output_root / "mendeley_ordinal_zero_shot_thresholds.csv", threshold_rows)

    main_threshold = min(thresholds)
    main_report = reports[f"{main_threshold:.3f}"]
    write_csv(args.output_root / "mendeley_ordinal_zero_shot_classwise.csv", main_report["classes"])
    write_confusion_matrix(args.output_root / "mendeley_ordinal_zero_shot_confusion_matrix.csv", main_report["confusion"])
    duplicate_summary = duplicate_report(
        third_domain_images=[record["image_path"] for record in records],
        reference_roots=args.duplicate_reference,
        output_csv=args.output_root / "mendeley_ordinal_duplicate_report.csv",
        hash_distance=args.duplicate_hash_distance,
    )
    summary = {
        "protocol": "strict_third_domain_zero_shot_image_level_classification",
        "dataset": "An Ordinal Dataset for Ripeness Level Classification in Oil Palm Fruit Quality Grading",
        "dataset_doi": "10.17632/424y96m6sw.1",
        "dataset_root": str(args.dataset_root),
        "official_split": "test",
        "evaluated_classes": list(MENDELEY_CLASSES.values()),
        "prediction_mapping": {
            "abnormal": "Decayed",
            "overripe": "OverRipe",
            "ripe": "FullyRipe",
            "under_ripe": "PartiallyRipe",
            "unripe": "Immature",
            "empty": "ignored_unmapped",
        },
        "important_note": "Image-level classification protocol. Do not compare directly with bounding-box mAP.",
        "model": str(args.model),
        "main_threshold": main_threshold,
        **main_report["summary"],
        "duplicate_check": duplicate_summary,
    }
    (args.output_root / "mendeley_ordinal_zero_shot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

