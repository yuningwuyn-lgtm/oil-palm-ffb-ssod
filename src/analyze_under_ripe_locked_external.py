"""Generate locked-external class-wise and under-ripe error analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from evaluate_locked_external_protocol import collect
from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    BoxPrediction,
    box_iou,
    evaluate_model_masked_classes,
    list_images,
)


UNDER_RIPE_ID = 4
BACKGROUND = "background"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_thresholds(summary_path: Path) -> dict[int, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        int(class_id): float(threshold)
        for class_id, threshold in payload["calibration_classwise_thresholds"].items()
    }


def xyxy(box: BoxPrediction, width: int, height: int) -> tuple[int, int, int, int]:
    left = int((box.x_center - box.width / 2) * width)
    top = int((box.y_center - box.height / 2) * height)
    right = int((box.x_center + box.width / 2) * width)
    bottom = int((box.y_center + box.height / 2) * height)
    return left, top, right, bottom


def annotate_example(
    image_path: Path,
    gt_boxes: list[BoxPrediction],
    pred_boxes: list[BoxPrediction],
    output_path: Path,
) -> None:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for box in gt_boxes:
            draw.rectangle(xyxy(box, width, height), outline="lime", width=4)
            draw.text(
                (xyxy(box, width, height)[0], max(0, xyxy(box, width, height)[1] - 14)),
                f"GT {CANONICAL_CLASSES[box.class_id]}",
                fill="lime",
            )
        for box in pred_boxes:
            draw.rectangle(xyxy(box, width, height), outline="red", width=3)
            draw.text(
                (xyxy(box, width, height)[0], xyxy(box, width, height)[1]),
                f"P {CANONICAL_CLASSES[box.class_id]} {box.confidence:.2f}",
                fill="red",
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def match_image(
    image_name: str,
    gt_boxes: list[BoxPrediction],
    pred_boxes: list[BoxPrediction],
    image_path: Path,
    confusion: Counter,
    error_rows: list[dict],
    example_dir: Path,
    max_examples: int,
    saved_counts: Counter,
) -> None:
    matched_gt: set[int] = set()
    pred_boxes = sorted(pred_boxes, key=lambda box: box.confidence, reverse=True)
    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx
        if best_idx >= 0 and best_iou >= 0.50:
            gt = gt_boxes[best_idx]
            matched_gt.add(best_idx)
            confusion[(gt.class_id, pred.class_id)] += 1
            if gt.class_id == UNDER_RIPE_ID and pred.class_id != UNDER_RIPE_ID:
                kind = "under_ripe_misclassified"
                error_rows.append(
                    {
                        "image": image_name,
                        "error_type": kind,
                        "gt_class": CANONICAL_CLASSES[gt.class_id],
                        "pred_class": CANONICAL_CLASSES[pred.class_id],
                        "confidence": pred.confidence,
                        "iou": best_iou,
                    }
                )
                if saved_counts[kind] < max_examples:
                    annotate_example(
                        image_path, gt_boxes, pred_boxes, example_dir / kind / image_path.name
                    )
                    saved_counts[kind] += 1
        else:
            confusion[(BACKGROUND, pred.class_id)] += 1
            if pred.class_id == UNDER_RIPE_ID:
                kind = "under_ripe_false_positive"
                error_rows.append(
                    {
                        "image": image_name,
                        "error_type": kind,
                        "gt_class": BACKGROUND,
                        "pred_class": CANONICAL_CLASSES[pred.class_id],
                        "confidence": pred.confidence,
                        "iou": best_iou,
                    }
                )
                if saved_counts[kind] < max_examples:
                    annotate_example(
                        image_path, gt_boxes, pred_boxes, example_dir / kind / image_path.name
                    )
                    saved_counts[kind] += 1
    for gt_idx, gt in enumerate(gt_boxes):
        if gt_idx in matched_gt:
            continue
        confusion[(gt.class_id, BACKGROUND)] += 1
        if gt.class_id == UNDER_RIPE_ID:
            kind = "under_ripe_false_negative"
            error_rows.append(
                {
                    "image": image_name,
                    "error_type": kind,
                    "gt_class": CANONICAL_CLASSES[gt.class_id],
                    "pred_class": BACKGROUND,
                    "confidence": 0.0,
                    "iou": 0.0,
                }
            )
            if saved_counts[kind] < max_examples:
                annotate_example(
                    image_path, gt_boxes, pred_boxes, example_dir / kind / image_path.name
                )
                saved_counts[kind] += 1


def write_confusion(output_root: Path, confusion: Counter) -> None:
    classes: list[int | str] = [*PREPROCESSED_FFB_EVAL_CLASS_IDS, BACKGROUND]
    names = {
        **{class_id: CANONICAL_CLASSES[class_id] for class_id in PREPROCESSED_FFB_EVAL_CLASS_IDS},
        BACKGROUND: BACKGROUND,
    }
    matrix = np.array(
        [[confusion[(gt_id, pred_id)] for pred_id in classes] for gt_id in classes],
        dtype=int,
    )
    rows = []
    for row_idx, gt_id in enumerate(classes):
        row = {"true\\pred": names[gt_id]}
        row.update({names[pred_id]: int(matrix[row_idx, col_idx]) for col_idx, pred_id in enumerate(classes)})
        rows.append(row)
    write_csv(output_root / "locked_external_confusion_matrix.csv", rows)
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    labels = [names[class_id] for class_id in classes]
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_root / "locked_external_confusion_matrix.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--locked-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.locked_summary)
    classwise_csv = args.output_root / "locked_external_classwise_ap.csv"
    evaluate_model_masked_classes(
        args.model,
        args.dataset_root,
        split="locked_final_test",
        allowed_class_ids=PREPROCESSED_FFB_EVAL_CLASS_IDS,
        args=SimpleNamespace(
            imgsz=args.imgsz,
            device=args.device,
            skip_threshold_search=True,
        ),
        output_csv=classwise_csv,
        calibration_csv=None,
    )
    model = YOLO(str(args.model))
    gt, pred = collect(model, args.dataset_root, "locked_final_test", args.imgsz, args.device)
    image_dir = args.dataset_root / "locked_final_test" / "images"
    image_by_name = {image.name: image for image in list_images(image_dir)}
    confusion: Counter = Counter()
    error_rows: list[dict] = []
    saved_counts: Counter = Counter()
    for image_name, gt_boxes in gt.items():
        filtered_pred = [
            box for box in pred[image_name] if box.confidence >= thresholds[box.class_id]
        ]
        match_image(
            image_name,
            gt_boxes,
            filtered_pred,
            image_by_name[image_name],
            confusion,
            error_rows,
            args.output_root / "under_ripe_examples",
            args.max_examples,
            saved_counts,
        )
    write_confusion(args.output_root, confusion)
    write_csv(
        args.output_root / "under_ripe_error_manifest.csv",
        error_rows,
        ["image", "error_type", "gt_class", "pred_class", "confidence", "iou"],
    )
    summary = {
        "protocol": "locked_external_posthoc_error_analysis_no_retuning",
        "model": str(args.model),
        "dataset_root": str(args.dataset_root),
        "threshold_source": str(args.locked_summary),
        "classwise_thresholds": thresholds,
        "under_ripe_error_counts": dict(Counter(row["error_type"] for row in error_rows)),
        "saved_examples": dict(saved_counts),
    }
    (args.output_root / "under_ripe_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

