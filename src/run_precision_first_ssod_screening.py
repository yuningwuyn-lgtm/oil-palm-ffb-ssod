"""Screen precision-first SSOD variants on calibration and source-valid only."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from full_ssod_ffb_pipeline import ensure_dir, evaluate_model


SOURCE_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\protocol_repair_v2\science_oilpalm_scene_coverage_disjoint")
EXTERNAL_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\protocol_repair_v2\external_preprocessed_ffb_calibration_locked")
SUPERVISED_ROOT = Path(r"LOCAL_DATA_ROOT\paper_protocol_v2_formal\datasets\model2_balanced")
NEW_IMAGES_ROOT = Path(r"LOCAL_DATA_ROOT\new_images_raw_ffb_augmented_for_ssod")
TEACHER = Path(r"LOCAL_DATA_ROOT\paper_protocol_v2_formal\seed_42\model2_balanced\runs\formal_train\weights\best.pt")
OUTPUT_ROOT = Path(r"LOCAL_DATA_ROOT\precision_first_ssod_screening")
REPORT_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\reports\precision_first_ssod_screening")


def trainer_args(imgsz: int) -> SimpleNamespace:
    return SimpleNamespace(imgsz=imgsz, device="0")


def checkpoint_for(variant: str) -> Path:
    if variant == "A_model2_balanced":
        return TEACHER
    return OUTPUT_ROOT / variant / "runs" / "self_train_iter_1" / "weights" / "best.pt"


def run_ssod_variant(variant: str, epochs: int, imgsz: int, batch: int) -> Path:
    checkpoint = checkpoint_for(variant)
    if checkpoint.exists():
        return checkpoint
    command = [
        sys.executable,
        str(Path(__file__).with_name("full_ssod_ffb_pipeline.py")),
        "--prepared-supervised-root", str(SUPERVISED_ROOT),
        "--new-images", str(NEW_IMAGES_ROOT),
        "--output-root", str(OUTPUT_ROOT / variant),
        "--skip-baseline-training",
        "--baseline-model-path", str(TEACHER),
        "--teacher-model", str(TEACHER),
        "--skip-protocol-reports",
        "--self-training-iterations", "1",
        "--max-epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--device", "0",
        "--train-cache", "disk",
        "--seed", "42",
        "--base-conf-threshold", "0.65",
        "--quality-threshold", "0.80",
        "--consistency-iou", "0.70",
        "--min-consistency-support", "3",
        "--num-aug-views", "4",
        "--edge-margin", "0.01",
        "--reject-edge-contact",
        "--max-boxes-per-image", "3",
        "--scene-quota-per-iteration", "100",
        "--max-pseudo-images-per-class", "100",
        "--enable-source-aware-loss-mask",
    ]
    if variant in {"C_strict_limited_relabel", "D_strict_relabel_box_quota"}:
        command.extend(
            [
                "--enable-folder-guided-hard-class-relabel",
                "--hard-classes", "abnormal,under_ripe",
                "--hard-class-source-map", "under_ripe:unripe|ripe,abnormal:empty|unripe|overripe",
                "--hard-class-min-quality", "0.85",
                "--hard-class-min-conf", "0.70",
                "--hard-class-min-support", "3",
            ]
        )
    if variant == "D_strict_relabel_box_quota":
        command.extend(
            [
                "--max-pseudo-boxes-per-class", "150",
                "--max-pseudo-boxes-per-class-overrides", "abnormal=80,under_ripe=80",
            ]
        )
    subprocess.run(command, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing candidate checkpoint: {checkpoint}")
    return checkpoint


def evaluate_candidate(variant: str, checkpoint: Path, imgsz: int) -> dict:
    output_root = REPORT_ROOT / variant
    ensure_dir(output_root)
    calibration_json = output_root / "calibration_only" / "calibration_only_summary.json"
    if not calibration_json.exists():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("evaluate_calibration_only_protocol.py")),
                "--model", str(checkpoint),
                "--dataset-root", str(EXTERNAL_ROOT),
                "--output-root", str(output_root / "calibration_only"),
                "--imgsz", str(imgsz),
            ],
            check=True,
        )
    calibration = json.loads(calibration_json.read_text(encoding="utf-8"))
    source_csv = output_root / "source_valid_metrics.csv"
    if source_csv.exists():
        source_row = next(csv.DictReader(source_csv.open(encoding="utf-8-sig")))
    else:
        metrics = evaluate_model(checkpoint, SOURCE_ROOT / "data.yaml", "val", trainer_args(imgsz))
        source_row = {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map5095": float(metrics.box.map),
        }
        with source_csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(source_row))
            writer.writeheader()
            writer.writerow(source_row)
    return {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "calibration_map50": float(calibration["calibration_detection_ap"]["map50"]),
        "calibration_map5095": float(calibration["calibration_detection_ap"]["map5095"]),
        "calibration_precision": float(calibration["calibration_classwise_metrics"]["macro_precision"]),
        "calibration_recall": float(calibration["calibration_classwise_metrics"]["macro_recall"]),
        "calibration_f1": float(calibration["calibration_classwise_metrics"]["macro_f1"]),
        "source_valid_map50": float(source_row["map50"]),
        "source_valid_map5095": float(source_row["map5095"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="A_model2_balanced,B_strict_no_relabel,C_strict_limited_relabel,D_strict_relabel_box_quota")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(REPORT_ROOT)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    rows = []
    for variant in variants:
        print(f"\n=== precision-first screening: {variant} ===")
        checkpoint = checkpoint_for(variant) if variant == "A_model2_balanced" else run_ssod_variant(variant, args.epochs, args.imgsz, args.batch)
        rows.append(evaluate_candidate(variant, checkpoint, args.imgsz))
    baseline = next(row for row in rows if row["variant"] == "A_model2_balanced")
    for row in rows:
        row["delta_calibration_map50_vs_A"] = row["calibration_map50"] - baseline["calibration_map50"]
        row["delta_source_valid_map50_vs_A"] = row["source_valid_map50"] - baseline["source_valid_map50"]
        row["delta_precision_vs_A"] = row["calibration_precision"] - baseline["calibration_precision"]
        row["guardrail_pass"] = (
            row["calibration_map50"] >= baseline["calibration_map50"]
            and row["source_valid_map50"] >= baseline["source_valid_map50"] - 0.02
            and row["calibration_precision"] >= baseline["calibration_precision"]
        )
    with (REPORT_ROOT / "precision_first_screening_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (REPORT_ROOT / "precision_first_screening_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

