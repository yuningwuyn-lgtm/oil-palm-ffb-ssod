"""Create high-precision comparison tables and qualitative figures.

The report compares the fair mixed-supervised baseline against the final
precision-oriented model on the same external 4-class masked test set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    BoxPrediction,
    box_to_pixels,
    ensure_dir,
    list_images,
    load_label_boxes,
    predict_one,
)


CLASS_COLORS = {
    0: "red",
    3: "orange",
    4: "cyan",
    5: "lime",
}


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def selected_thresholds(path: Path) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    for row in read_csv(path):
        if row.get("selected") != "1":
            continue
        if not row.get("class_id"):
            continue
        thresholds[int(float(row["class_id"]))] = float(row["threshold"])
    return thresholds


def write_high_precision_table(runs: Sequence[Tuple[str, Path]], output_csv: Path) -> None:
    rows: List[dict] = []
    for name, report_dir in runs:
        metrics = read_csv(report_dir / "external_preprocessed_ffb_metrics.csv")
        macro = next(row for row in metrics if row.get("class_name") == "macro")
        rows.append(
            {
                "model": name,
                "map50": macro.get("ap50", ""),
                "map50_95": macro.get("ap50_95", ""),
                "calibrated_precision": macro.get("best_calibrated_precision", ""),
                "calibrated_recall": macro.get("best_calibrated_recall", ""),
                "calibrated_f1": macro.get("best_calibrated_f1", ""),
                "high_precision_precision": macro.get("classwise_threshold_precision", ""),
                "high_precision_recall": macro.get("classwise_threshold_recall", ""),
                "high_precision_f1": macro.get("classwise_threshold_f1", ""),
                "classwise_thresholds": macro.get("classwise_thresholds", ""),
            }
        )
    ensure_dir(output_csv.parent)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def plot_pr_curves(runs: Sequence[Tuple[str, Path]], output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, report_dir in runs:
        df = pd.read_csv(report_dir / "external_preprocessed_ffb_metrics_pr_curve.csv")
        macro = (
            df.groupby("rank", as_index=False)[["precision", "recall"]]
            .mean(numeric_only=True)
            .sort_values("recall")
        )
        plt.plot(macro["recall"], macro["precision"], label=name, linewidth=2)
    plt.axhline(0.85, color="black", linestyle="--", linewidth=1, label="P=0.85 target")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("External 4-class PR curve")
    plt.grid(True, alpha=0.25)
    plt.legend()
    ensure_dir(output_path.parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def draw_panel(image_path: Path, label_path: Path, predictions: Sequence[BoxPrediction], thresholds: Dict[int, float], title: str) -> Image.Image:
    with Image.open(image_path) as image:
        panel = image.convert("RGB")
    draw = ImageDraw.Draw(panel)
    for gt in load_label_boxes(label_path, allowed_class_ids=set(PREPROCESSED_FFB_EVAL_CLASS_IDS)):
        draw.rectangle(box_to_pixels(gt, panel.width, panel.height), outline="white", width=3)
    for pred in predictions:
        threshold = thresholds.get(pred.class_id, 1.0)
        if pred.confidence < threshold:
            continue
        color = CLASS_COLORS.get(pred.class_id, "yellow")
        x1, y1, x2, y2 = box_to_pixels(pred, panel.width, panel.height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, max(0, y1 - 14)), f"{CANONICAL_CLASSES[pred.class_id]} {pred.confidence:.2f}", fill=color)
    draw.rectangle((0, 0, panel.width, 24), fill=(0, 0, 0))
    draw.text((6, 5), title, fill="white")
    return panel


def make_qualitative_pairs(
    mixed_model_path: Path,
    final_model_path: Path,
    mixed_report_dir: Path,
    final_report_dir: Path,
    external_test_root: Path,
    output_dir: Path,
    imgsz: int,
    device: str,
    max_images: int,
) -> None:
    ensure_dir(output_dir)
    mixed_thresholds = selected_thresholds(mixed_report_dir / "external_preprocessed_ffb_metrics_classwise_threshold_search.csv")
    final_thresholds = selected_thresholds(final_report_dir / "external_preprocessed_ffb_metrics_classwise_threshold_search.csv")
    mixed_fp = read_csv(mixed_report_dir / "external_preprocessed_ffb_metrics_false_positive_examples.csv")
    candidate_names = [row["image"] for row in mixed_fp if row.get("image")]
    image_dir = external_test_root / "test" / "images"
    label_dir = external_test_root / "test" / "labels"
    images = {path.name: path for path in list_images(image_dir)}
    selected = [images[name] for name in candidate_names if name in images][:max_images]
    if not selected:
        selected = list_images(image_dir)[:max_images]

    mixed_model = YOLO(str(mixed_model_path))
    final_model = YOLO(str(final_model_path))
    for idx, image_path in enumerate(selected, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"
        mixed_preds = predict_one(mixed_model, image_path, imgsz=imgsz, device=device, conf=0.05)
        final_preds = predict_one(final_model, image_path, imgsz=imgsz, device=device, conf=0.05)
        left = draw_panel(image_path, label_path, mixed_preds, mixed_thresholds, "Mixed supervised baseline")
        right = draw_panel(image_path, label_path, final_preds, final_thresholds, "Final SSOD + FP250 + URpos")
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "black")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        canvas.save(output_dir / f"fp_reduction_{idx:03d}_{image_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-precision report artifacts.")
    parser.add_argument("--mixed-report-dir", type=Path, required=True)
    parser.add_argument("--final-report-dir", type=Path, required=True)
    parser.add_argument("--mixed-model", type=Path, required=True)
    parser.add_argument("--final-model", type=Path, required=True)
    parser.add_argument("--external-test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-qualitative-images", type=int, default=24)
    args = parser.parse_args()

    runs = [
        ("Mixed supervised 640", args.mixed_report_dir),
        ("Final SSOD FP250 URpos 640", args.final_report_dir),
    ]
    ensure_dir(args.output_dir)
    write_high_precision_table(runs, args.output_dir / "high_precision_operating_points.csv")
    plot_pr_curves(runs, args.output_dir / "external_pr_curve_mixed_vs_final.png")
    make_qualitative_pairs(
        mixed_model_path=args.mixed_model,
        final_model_path=args.final_model,
        mixed_report_dir=args.mixed_report_dir,
        final_report_dir=args.final_report_dir,
        external_test_root=args.external_test_root,
        output_dir=args.output_dir / "qualitative_fp_reduction_pairs",
        imgsz=args.imgsz,
        device=args.device,
        max_images=args.max_qualitative_images,
    )
    print(f"Wrote report artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()

