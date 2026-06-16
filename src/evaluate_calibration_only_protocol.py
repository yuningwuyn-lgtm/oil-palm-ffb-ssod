"""Evaluate SSOD screening candidates on calibration data without opening locked-final."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from ultralytics import YOLO

from evaluate_locked_external_protocol import collect, evaluate_thresholds
from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    evaluate_model_masked_classes,
    write_classwise_threshold_search_table,
    write_confidence_calibration_table,
    write_rows_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--target-precision", type=float, default=0.85)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    gt, pred = collect(model, args.dataset_root, "calibration", args.imgsz, args.device)
    threshold_grid = [0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    global_fit = write_confidence_calibration_table(
        args.output_root / "calibration_global_threshold_search.csv",
        gt,
        pred,
        PREPROCESSED_FFB_EVAL_CLASS_IDS,
        threshold_grid,
        args.model,
        args.dataset_root,
    )
    classwise_fit = write_classwise_threshold_search_table(
        args.output_root / "calibration_classwise_threshold_search.csv",
        gt,
        pred,
        PREPROCESSED_FFB_EVAL_CLASS_IDS,
        threshold_grid,
        args.model,
        args.dataset_root,
        args.target_precision,
    )
    thresholds = {
        class_id: float(classwise_fit["selected_thresholds"][CANONICAL_CLASSES[class_id]])
        for class_id in PREPROCESSED_FFB_EVAL_CLASS_IDS
    }
    classwise_metrics = evaluate_thresholds(gt, pred, thresholds)
    write_rows_csv(args.output_root / "calibration_classwise_metrics.csv", classwise_metrics["classes"])
    ap = evaluate_model_masked_classes(
        args.model,
        args.dataset_root,
        split="calibration",
        allowed_class_ids=PREPROCESSED_FFB_EVAL_CLASS_IDS,
        args=SimpleNamespace(imgsz=args.imgsz, device=args.device),
        output_csv=None,
        calibration_csv=None,
    )
    payload = {
        "protocol": "calibration_only_screening_no_locked_final_access",
        "model": str(args.model),
        "dataset_root": str(args.dataset_root),
        "calibration_images": len(gt),
        "calibration_global_threshold": global_fit,
        "calibration_classwise_thresholds": thresholds,
        "calibration_classwise_metrics": classwise_metrics,
        "calibration_detection_ap": {"map50": ap["map50"], "map5095": ap["map5095"]},
    }
    (args.output_root / "calibration_only_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

