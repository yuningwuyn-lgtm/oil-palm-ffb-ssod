"""Generate a 4-class external masked confusion matrix at a fixed threshold."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    box_iou,
    convert_to_yolo_format,
    create_masked_external_dataset,
    list_images,
    load_label_boxes,
    result_to_boxes,
    set_safe_delete_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="External 4-class confusion matrix.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("LOCAL_DATA_ROOT/preprocessed-ffb/yolov8"))
    parser.add_argument("--output-dir", type=Path, default=Path("LOCAL_PROJECT_ROOT/paper_framework/reports/paper_summary"))
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.output_dir / "external_confusion_dataset"
    set_safe_delete_root(work_root)
    canonical = convert_to_yolo_format(args.dataset, work_root / "canonical")
    masked = create_masked_external_dataset(canonical.parent, work_root / "masked", PREPROCESSED_FFB_EVAL_CLASS_IDS)

    classes = list(PREPROCESSED_FFB_EVAL_CLASS_IDS)
    class_to_idx = {class_id: idx for idx, class_id in enumerate(classes)}
    labels = [CANONICAL_CLASSES[class_id] for class_id in classes]
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    unmatched_gt = np.zeros(len(classes), dtype=int)
    false_positive = np.zeros(len(classes), dtype=int)

    model = YOLO(str(args.model))
    images = list_images(masked.parent / "test" / "images")
    label_dir = masked.parent / "test" / "labels"
    for image_path in images:
        gt_boxes = load_label_boxes(label_dir / f"{image_path.stem}.txt", set(classes))
        results = model.predict(source=str(image_path), imgsz=args.imgsz, device=args.device, conf=args.threshold, save=False, verbose=False)
        pred_boxes = [box for box in result_to_boxes(results[0]) if box.class_id in class_to_idx] if results else []
        pred_boxes.sort(key=lambda box: box.confidence, reverse=True)
        matched_gt = set()
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
            if best_iou >= 0.50 and best_idx >= 0:
                matched_gt.add(best_idx)
                matrix[class_to_idx[gt_boxes[best_idx].class_id], class_to_idx[pred.class_id]] += 1
            else:
                false_positive[class_to_idx[pred.class_id]] += 1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                unmatched_gt[class_to_idx[gt.class_id]] += 1

    csv_path = args.output_dir / "model2_external_confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *labels, "unmatched_gt"])
        for idx, label in enumerate(labels):
            writer.writerow([label, *matrix[idx].tolist(), int(unmatched_gt[idx])])
        writer.writerow(["false_positive", *false_positive.tolist(), ""])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_path = args.output_dir / "model2_external_confusion_matrix.png"
    plt.savefig(fig_path, dpi=220)
    plt.close()
    print(f"Wrote confusion matrix: {csv_path}")
    print(f"Wrote confusion matrix plot: {fig_path}")


if __name__ == "__main__":
    main()

