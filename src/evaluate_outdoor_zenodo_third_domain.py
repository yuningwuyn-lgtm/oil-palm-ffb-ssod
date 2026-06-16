"""
Strict zero-shot third-domain evaluation for the 2024 Outdoor Tenera FFB dataset.

This dataset is image-level classification data, not a bounding-box benchmark.
It must remain evaluation-only: do not merge it into training, pseudo-labeling,
hard-negative mining, or threshold fitting. The official FFBtest split is used
as the paper-grade result. See DOI: 10.5281/zenodo.11114885.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ultralytics import YOLO

from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    average_image_hash,
    download_zenodo_record,
    hamming_distance,
    list_images,
    normalized_base_filename,
    predict_one,
)


ZENODO_RECORD_ID = "11114885"
OUTDOOR_CLASS_IDS = (0, 1, 2, 3, 5)
OUTDOOR_CLASS_NAMES = {class_id: CANONICAL_CLASSES[class_id] for class_id in OUTDOOR_CLASS_IDS}
PREDICTION_CLASS_REMAP = {4: 5}  # Outdoor labels combine under-ripe with unripe.
LABEL_PATTERNS = {
    "damaged": 0,
    "empty": 1,
    "overripe": 2,
    "unripe": 5,
    "ripe": 3,
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


def infer_outdoor_label(image_path: Path) -> Optional[int]:
    text = "_".join(part.lower() for part in image_path.parts[-4:])
    for class_name, class_id in LABEL_PATTERNS.items():
        if class_name in text:
            return class_id
    return None


def infer_official_split(image_path: Path) -> str:
    text = "_".join(part.lower() for part in image_path.parts[-4:])
    if "ffbtest" in text or re.search(r"(?<![a-z])test", text):
        return "test"
    if "ffbtrain" in text or re.search(r"(?<![a-z])train", text):
        return "train"
    return "unknown"


def discover_dataset_root(extracted_dirs: Sequence[Path]) -> Path:
    candidates = []
    for extracted_dir in extracted_dirs:
        for directory in [extracted_dir, *extracted_dir.rglob("*")]:
            if not directory.is_dir():
                continue
            images = list_images(directory)
            if images:
                labeled = sum(infer_outdoor_label(image) is not None for image in images)
                candidates.append((labeled, directory))
    if not candidates:
        raise FileNotFoundError("No Outdoor Tenera FFB images were found after extraction.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def collect_records(dataset_root: Path, split: str) -> List[dict]:
    records = []
    for image_path in list_images(dataset_root):
        label_id = infer_outdoor_label(image_path)
        official_split = infer_official_split(image_path)
        if label_id is None or (split != "all" and official_split != split):
            continue
        records.append(
            {
                "image_path": image_path,
                "official_split": official_split,
                "gt_class_id": label_id,
                "gt_class": OUTDOOR_CLASS_NAMES[label_id],
            }
        )
    return records


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_predictions(rows: Sequence[dict], threshold: float) -> dict:
    confusion = Counter()
    gt_counts = Counter()
    predicted_counts = Counter()
    detected = 0
    correct = 0
    for row in rows:
        gt_id = int(row["gt_class_id"])
        pred_id = row["pred_class_id"]
        confidence = float(row["confidence"])
        gt_counts[gt_id] += 1
        if pred_id == "" or confidence < threshold:
            pred_id = "no_detection"
        else:
            pred_id = int(pred_id)
            detected += 1
            predicted_counts[pred_id] += 1
            if pred_id == gt_id:
                correct += 1
        confusion[(gt_id, pred_id)] += 1

    class_metrics = []
    for class_id in OUTDOOR_CLASS_IDS:
        tp = confusion[(class_id, class_id)]
        fp = sum(confusion[(other_id, class_id)] for other_id in OUTDOOR_CLASS_IDS if other_id != class_id)
        fn = sum(count for (gt_id, pred_id), count in confusion.items() if gt_id == class_id and pred_id != class_id)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        class_metrics.append(
            {
                "class_id": class_id,
                "class_name": OUTDOOR_CLASS_NAMES[class_id],
                "gt_images": gt_counts[class_id],
                "predicted_images": predicted_counts[class_id],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": safe_div(2 * precision * recall, precision + recall),
            }
        )

    macro_precision = sum(row["precision"] for row in class_metrics) / len(class_metrics)
    macro_recall = sum(row["recall"] for row in class_metrics) / len(class_metrics)
    macro_f1 = sum(row["f1"] for row in class_metrics) / len(class_metrics)
    summary = {
        "threshold": threshold,
        "images": len(rows),
        "detected_images": detected,
        "coverage": safe_div(detected, len(rows)),
        "top1_accuracy": safe_div(correct, len(rows)),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
    return {"summary": summary, "classes": class_metrics, "confusion": confusion}


def evaluate_zero_shot(
    model_path: Path,
    records: Sequence[dict],
    imgsz: int,
    device: str,
    min_prediction_conf: float,
) -> List[dict]:
    model = YOLO(str(model_path))
    rows = []
    for index, record in enumerate(records, start=1):
        predictions = predict_one(model, record["image_path"], imgsz=imgsz, device=device, conf=min_prediction_conf)
        best = max(predictions, key=lambda box: box.confidence, default=None)
        raw_id = best.class_id if best is not None else ""
        mapped_id = PREDICTION_CLASS_REMAP.get(raw_id, raw_id) if best is not None else ""
        rows.append(
            {
                "image": str(record["image_path"]),
                "official_split": record["official_split"],
                "gt_class_id": record["gt_class_id"],
                "gt_class": record["gt_class"],
                "raw_pred_class_id": raw_id,
                "raw_pred_class": CANONICAL_CLASSES.get(raw_id, "no_detection") if best is not None else "no_detection",
                "pred_class_id": mapped_id,
                "pred_class": OUTDOOR_CLASS_NAMES.get(mapped_id, "no_detection") if best is not None else "no_detection",
                "confidence": f"{best.confidence:.6f}" if best is not None else "0.000000",
                "box_x": f"{best.x_center:.6f}" if best is not None else "",
                "box_y": f"{best.y_center:.6f}" if best is not None else "",
                "box_w": f"{best.width:.6f}" if best is not None else "",
                "box_h": f"{best.height:.6f}" if best is not None else "",
            }
        )
        if index % 20 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)} images")
    return rows


def write_confusion_matrix(path: Path, confusion: Counter) -> None:
    columns: List[int | str] = [*OUTDOOR_CLASS_IDS, "no_detection"]
    rows = []
    for gt_id in OUTDOOR_CLASS_IDS:
        row = {"gt_class": OUTDOOR_CLASS_NAMES[gt_id]}
        for pred_id in columns:
            pred_name = OUTDOOR_CLASS_NAMES.get(pred_id, "no_detection")
            row[f"pred_{pred_name}"] = confusion[(gt_id, pred_id)]
        rows.append(row)
    write_csv(path, rows)


def duplicate_report(
    third_domain_images: Sequence[Path],
    reference_roots: Sequence[Path],
    output_csv: Path,
    hash_distance: int,
) -> dict:
    third_keys = defaultdict(list)
    third_hashes = defaultdict(list)
    for image in third_domain_images:
        third_keys[normalized_base_filename(image)].append(image)
        image_hash = average_image_hash(image)
        if image_hash is not None:
            third_hashes[image_hash].append(image)

    rows = []
    reference_images = 0
    for root in reference_roots:
        if not root.exists():
            continue
        for reference in list_images(root):
            reference_images += 1
            key = normalized_base_filename(reference)
            for third_image in third_keys.get(key, []):
                rows.append(
                    {
                        "match_type": "normalized_filename",
                        "distance": 0,
                        "third_domain_image": str(third_image),
                        "reference_image": str(reference),
                        "reference_root": str(root),
                    }
                )
            reference_hash = average_image_hash(reference)
            if reference_hash is None:
                continue
            for third_hash, third_images in third_hashes.items():
                distance = hamming_distance(reference_hash, third_hash)
                if distance <= hash_distance:
                    for third_image in third_images:
                        rows.append(
                            {
                                "match_type": "average_hash",
                                "distance": distance,
                                "third_domain_image": str(third_image),
                                "reference_image": str(reference),
                                "reference_root": str(root),
                            }
                        )
    unique_rows = list({tuple(row.items()): row for row in rows}.values())
    write_csv(
        output_csv,
        unique_rows,
        fieldnames=["match_type", "distance", "third_domain_image", "reference_image", "reference_root"],
    )
    return {
        "third_domain_images": len(third_domain_images),
        "reference_images": reference_images,
        "duplicate_matches": len(unique_rows),
        "hash_distance": hash_distance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict zero-shot evaluation on the 2024 Outdoor Tenera FFB third domain.")
    parser.add_argument("--download-root", type=Path, default=Path(r"LOCAL_DATA_ROOT\public_datasets\zenodo_outdoor_tenera_ffb"))
    parser.add_argument("--dataset-root", type=Path, default=None, help="Use an already extracted dataset and skip Zenodo download.")
    parser.add_argument("--model", type=Path, default=Path(r"LOCAL_DATA_ROOT\paper_stage2_formal\seed_42\model2_ssod_fp250_urpos640_yolov8n\runs\baseline_supervised\weights\best.pt"))
    parser.add_argument("--output-root", type=Path, default=Path(r"LOCAL_PROJECT_ROOT\paper_framework\reports\third_domain_outdoor_zenodo"))
    parser.add_argument("--split", choices=("test", "all"), default="test")
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

    if args.dataset_root:
        dataset_root = args.dataset_root
    else:
        dataset_root = discover_dataset_root(download_zenodo_record(ZENODO_RECORD_ID, args.download_root))
    records = collect_records(dataset_root, args.split)
    if not records:
        raise ValueError(f"No labeled outdoor records found for split={args.split}: {dataset_root}")

    print(f"Outdoor third-domain root: {dataset_root}")
    print(f"Official split: {args.split}; images={len(records)}")
    print("This dataset is evaluation-only and will not be merged into any training source.")

    predictions = evaluate_zero_shot(
        model_path=args.model,
        records=records,
        imgsz=args.imgsz,
        device=args.device,
        min_prediction_conf=args.min_prediction_conf,
    )
    write_csv(args.output_root / "outdoor_zenodo_zero_shot_predictions.csv", predictions)

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    threshold_rows = []
    reports = {}
    for threshold in thresholds:
        report = summarize_predictions(predictions, threshold)
        threshold_rows.append(report["summary"])
        reports[f"{threshold:.3f}"] = report
    write_csv(args.output_root / "outdoor_zenodo_zero_shot_thresholds.csv", threshold_rows)

    main_threshold = min(thresholds)
    main_report = reports[f"{main_threshold:.3f}"]
    write_csv(args.output_root / "outdoor_zenodo_zero_shot_classwise.csv", main_report["classes"])
    write_confusion_matrix(args.output_root / "outdoor_zenodo_zero_shot_confusion_matrix.csv", main_report["confusion"])

    duplicate_summary = duplicate_report(
        third_domain_images=[record["image_path"] for record in records],
        reference_roots=args.duplicate_reference,
        output_csv=args.output_root / "outdoor_zenodo_duplicate_report.csv",
        hash_distance=args.duplicate_hash_distance,
    )
    summary = {
        "protocol": "strict_third_domain_zero_shot_image_level_classification",
        "dataset": "Outdoor Tenera Oil Palm Fruit Image: FFB Dataset",
        "dataset_doi": "10.5281/zenodo.11114885",
        "article_doi": "10.1016/j.dib.2024.110667",
        "dataset_root": str(dataset_root),
        "official_split": args.split,
        "evaluated_classes": list(OUTDOOR_CLASS_NAMES.values()),
        "prediction_remap": {"under_ripe": "unripe"},
        "important_note": "Image-level classification protocol. Do not compare directly with bounding-box mAP.",
        "model": str(args.model),
        "main_threshold": main_threshold,
        **main_report["summary"],
        "duplicate_check": duplicate_summary,
    }
    (args.output_root / "outdoor_zenodo_zero_shot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

