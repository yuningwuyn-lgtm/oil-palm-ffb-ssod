"""Calibrate thresholds on target calibration data and evaluate locked-final target data."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from ultralytics import YOLO

from full_ssod_ffb_pipeline import (
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    CANONICAL_CLASSES,
    evaluate_model_masked_classes,
    fixed_confidence_pr,
    list_images,
    load_label_boxes,
    predict_one,
    write_classwise_threshold_search_table,
    write_confidence_calibration_table,
    write_rows_csv,
)


def collect(model: YOLO, root: Path, split: str, imgsz: int, device: str):
    labels = root / split / "labels"
    images = list_images(root / split / "images")
    gt = {image.name: load_label_boxes(labels / f"{image.stem}.txt", set(PREPROCESSED_FFB_EVAL_CLASS_IDS)) for image in images}
    pred = {
        image.name: [
            box
            for box in predict_one(model, image, imgsz=imgsz, device=device, conf=0.001)
            if box.class_id in PREPROCESSED_FFB_EVAL_CLASS_IDS
        ]
        for image in images
    }
    return gt, pred


def evaluate_thresholds(gt, pred, thresholds: dict[int, float]) -> dict:
    rows = []
    for class_id in PREPROCESSED_FFB_EVAL_CLASS_IDS:
        precision, recall, f1, tp, fp, fn = fixed_confidence_pr(gt, pred, class_id, thresholds[class_id], 0.50)
        rows.append(
            {
                "class_id": class_id,
                "class_name": CANONICAL_CLASSES[class_id],
                "threshold": thresholds[class_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return {
        "classes": rows,
        "macro_precision": sum(row["precision"] for row in rows) / len(rows),
        "macro_recall": sum(row["recall"] for row in rows) / len(rows),
        "macro_f1": sum(row["f1"] for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--target-precision", type=float, default=0.85)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    calibration_gt, calibration_pred = collect(model, args.dataset_root, "calibration", args.imgsz, args.device)
    final_gt, final_pred = collect(model, args.dataset_root, "locked_final_test", args.imgsz, args.device)
    threshold_grid = [0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    global_fit = write_confidence_calibration_table(
        args.output_root / "calibration_global_threshold_search.csv",
        calibration_gt,
        calibration_pred,
        PREPROCESSED_FFB_EVAL_CLASS_IDS,
        threshold_grid,
        args.model,
        args.dataset_root,
    )
    classwise_fit = write_classwise_threshold_search_table(
        args.output_root / "calibration_classwise_threshold_search.csv",
        calibration_gt,
        calibration_pred,
        PREPROCESSED_FFB_EVAL_CLASS_IDS,
        threshold_grid,
        args.model,
        args.dataset_root,
        args.target_precision,
    )
    global_thresholds = {class_id: float(global_fit["threshold"]) for class_id in PREPROCESSED_FFB_EVAL_CLASS_IDS}
    classwise_thresholds = {
        class_id: float(classwise_fit["selected_thresholds"][CANONICAL_CLASSES[class_id]])
        for class_id in PREPROCESSED_FFB_EVAL_CLASS_IDS
    }
    global_final = evaluate_thresholds(final_gt, final_pred, global_thresholds)
    classwise_final = evaluate_thresholds(final_gt, final_pred, classwise_thresholds)
    write_rows_csv(args.output_root / "locked_final_global_threshold_metrics.csv", global_final["classes"])
    write_rows_csv(args.output_root / "locked_final_classwise_threshold_metrics.csv", classwise_final["classes"])
    ap_summary = evaluate_model_masked_classes(
        args.model,
        args.dataset_root,
        split="locked_final_test",
        allowed_class_ids=PREPROCESSED_FFB_EVAL_CLASS_IDS,
        args=SimpleNamespace(imgsz=args.imgsz, device=args.device),
        output_csv=None,
        calibration_csv=None,
    )
    payload = {
        "protocol": "calibration_fit_then_locked_final_evaluation",
        "model": str(args.model),
        "dataset_root": str(args.dataset_root),
        "calibration_images": len(calibration_gt),
        "locked_final_images": len(final_gt),
        "calibration_global_threshold": global_fit,
        "locked_final_global_threshold_metrics": global_final,
        "calibration_classwise_thresholds": classwise_thresholds,
        "locked_final_classwise_threshold_metrics": classwise_final,
        "locked_final_detection_ap": {
            "map50": ap_summary["map50"],
            "map5095": ap_summary["map5095"],
        },
        "historical_note": "The original target test was inspected before this protocol repair. Treat this as a prospectively locked final subset for all future experiments.",
    }
    (args.output_root / "locked_external_protocol_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

