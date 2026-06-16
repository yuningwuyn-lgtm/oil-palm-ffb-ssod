"""Run the protocol-v2 three-model, three-seed formal comparison queue."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from ultralytics import YOLO

from evaluate_locked_external_protocol import main as _unused_main
from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    copy_yolo_split,
    ensure_dir,
    evaluate_model,
    link_or_copy_file,
    merge_source_metadata,
    train_yolo,
    write_data_yaml,
)


SEEDS = (42, 2026, 3407)
SOURCE_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\protocol_repair_v2\science_oilpalm_scene_coverage_disjoint")
BALANCED_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\preprocessed_ffb_class_balanced\target_800\balanced")
URPOS_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\precision_target085_reduced_subsets\under_ripe_positive_subset")
FPHARD_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\precision_target085_reduced_subsets\false_positive_hard_reduced")
LOCKED_EXTERNAL_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\protocol_repair_v2\external_preprocessed_ffb_calibration_locked")
NEW_IMAGES_ROOT = Path(r"LOCAL_DATA_ROOT\new_images_raw_ffb_augmented_for_ssod")
OUTPUT_ROOT = Path(r"LOCAL_DATA_ROOT\paper_protocol_v2_formal")
REPORT_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\reports\protocol_v2_formal")


def write_metadata(root: Path, sources: list[Path]) -> None:
    merge_source_metadata(sources, root)


def build_dataset(output_root: Path, include_balanced: bool, include_refinement: bool) -> Path:
    """Build one leakage-safe training view with explicit source prefixes."""
    marker = output_root / ".complete"
    if marker.exists() and (output_root / "data.yaml").exists():
        return output_root / "data.yaml"
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "valid", "test"):
        ensure_dir(output_root / split / "images")
        ensure_dir(output_root / split / "labels")
    sources = [SOURCE_ROOT]
    for split in ("train", "valid", "test"):
        copy_yolo_split(SOURCE_ROOT, split, output_root, prefix="src1_")
    if include_balanced:
        sources.append(BALANCED_ROOT)
        copy_yolo_split(BALANCED_ROOT, "train", output_root, prefix="src2_")
        copy_yolo_split(BALANCED_ROOT, "valid", output_root, prefix="src2_")
    if include_refinement:
        sources.extend([URPOS_ROOT, FPHARD_ROOT])
        copy_yolo_split(URPOS_ROOT, "train", output_root, prefix="src3_")
        copy_yolo_split(FPHARD_ROOT, "train", output_root, prefix="src4_")
    write_data_yaml(output_root)
    write_metadata(output_root, sources)
    marker.write_text("ok\n", encoding="ascii")
    return output_root / "data.yaml"


def trainer_args(seed: int, epochs: int, imgsz: int, batch: int, source_aware: bool) -> SimpleNamespace:
    return SimpleNamespace(
        clean_runs=True,
        max_epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="0",
        train_cache="disk",
        seed=seed,
        amp=False,
        ordinal_loss_gain=0.0,
        enable_source_aware_loss_mask=source_aware,
    )


def write_internal_metrics(path: Path, metrics, stage: str, checkpoint: Path, data_yaml: Path) -> None:
    row = {
        "stage": stage,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map5095": float(metrics.box.map),
        "checkpoint": str(checkpoint),
        "data_yaml": str(data_yaml),
    }
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def locked_external_eval(checkpoint: Path, output_root: Path, imgsz: int) -> None:
    summary = output_root / "locked_external" / "locked_external_protocol_summary.json"
    if summary.exists():
        return
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_locked_external_protocol.py")),
        "--model",
        str(checkpoint),
        "--dataset-root",
        str(LOCKED_EXTERNAL_ROOT),
        "--output-root",
        str(output_root / "locked_external"),
        "--imgsz",
        str(imgsz),
    ]
    subprocess.run(command, check=True)


def stage_checkpoint(stage: str, seed: int) -> Path:
    return (
        OUTPUT_ROOT
        / f"seed_{seed}"
        / stage
        / "runs"
        / ("self_train_iter_1" if stage == "stage1_ssod_teacher" else "formal_train")
        / "weights"
        / "best.pt"
    )


def train_stage1_ssod_teacher(seed: int, epochs: int, imgsz: int, batch: int) -> Path:
    """Generate a protocol-v2 SSOD teacher from the freshly trained Model2 checkpoint."""
    checkpoint = stage_checkpoint("stage1_ssod_teacher", seed)
    if checkpoint.exists():
        return checkpoint
    model2_checkpoint = stage_checkpoint("model2_balanced", seed)
    if not model2_checkpoint.exists():
        raise FileNotFoundError(f"Missing protocol-v2 Model2 checkpoint for seed {seed}: {model2_checkpoint}")
    output_root = OUTPUT_ROOT / f"seed_{seed}" / "stage1_ssod_teacher"
    command = [
        sys.executable,
        str(Path(__file__).with_name("full_ssod_ffb_pipeline.py")),
        "--prepared-supervised-root",
        str(OUTPUT_ROOT / "datasets" / "model2_balanced"),
        "--new-images",
        str(NEW_IMAGES_ROOT),
        "--output-root",
        str(output_root),
        "--skip-baseline-training",
        "--baseline-model-path",
        str(model2_checkpoint),
        "--teacher-model",
        str(model2_checkpoint),
        "--skip-protocol-reports",
        "--self-training-iterations",
        "1",
        "--max-epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--device",
        "0",
        "--train-cache",
        "disk",
        "--seed",
        str(seed),
        "--quality-threshold",
        "0.72",
        "--base-conf-threshold",
        "0.55",
        "--scene-quota-per-iteration",
        "100",
        "--max-pseudo-images-per-class",
        "100",
        "--enable-folder-guided-hard-class-relabel",
        "--hard-classes",
        "abnormal,under_ripe",
        "--hard-class-source-map",
        "under_ripe:unripe|ripe,abnormal:empty|unripe|overripe",
        "--hard-class-min-quality",
        "0.80",
        "--hard-class-min-conf",
        "0.55",
        "--hard-class-min-support",
        "1",
        "--enable-source-aware-loss-mask",
    ]
    subprocess.run(command, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage-1 SSOD training did not create checkpoint: {checkpoint}")
    return checkpoint


def run_one(stage: str, seed: int, epochs: int, imgsz: int, batch: int) -> None:
    run_root = OUTPUT_ROOT / f"seed_{seed}" / stage
    report_root = REPORT_ROOT / f"seed_{seed}" / stage
    ensure_dir(run_root)
    ensure_dir(report_root)
    if stage == "source_only":
        data_yaml = build_dataset(OUTPUT_ROOT / "datasets" / "source_only", False, False)
        init_model = "yolov8n.pt"
        source_aware = False
    elif stage == "model2_balanced":
        data_yaml = build_dataset(OUTPUT_ROOT / "datasets" / "model2_balanced", True, False)
        init_model = "yolov8n.pt"
        source_aware = True
    elif stage == "stage1_ssod_teacher":
        data_yaml = SOURCE_ROOT / "data.yaml"
        checkpoint = train_stage1_ssod_teacher(seed, epochs, imgsz, batch)
        args = trainer_args(seed, epochs, imgsz, batch, True)
        internal_csv = report_root / "scene_disjoint_test_metrics.csv"
        if not internal_csv.exists():
            metrics = evaluate_model(checkpoint, data_yaml, "test", args)
            write_internal_metrics(internal_csv, metrics, f"{stage}_scene_disjoint_test", checkpoint, data_yaml)
        locked_external_eval(checkpoint, report_root, imgsz)
        return
    elif stage == "stage2_refined":
        data_yaml = build_dataset(OUTPUT_ROOT / "datasets" / "stage2_refined", True, True)
        teacher = stage_checkpoint("stage1_ssod_teacher", seed)
        if not teacher.exists():
            raise FileNotFoundError(f"Missing protocol-v2 Stage-1 SSOD teacher for seed {seed}: {teacher}")
        init_model = teacher
        source_aware = True
    else:
        raise ValueError(stage)

    checkpoint = run_root / "runs" / "formal_train" / "weights" / "best.pt"
    args = trainer_args(seed, epochs, imgsz, batch, source_aware)
    if not checkpoint.exists():
        checkpoint = train_yolo(init_model, data_yaml, run_root / "runs", "formal_train", args)
    internal_csv = report_root / "scene_disjoint_test_metrics.csv"
    if not internal_csv.exists():
        source_test_yaml = SOURCE_ROOT / "data.yaml"
        metrics = evaluate_model(checkpoint, source_test_yaml, "test", args)
        write_internal_metrics(internal_csv, metrics, f"{stage}_scene_disjoint_test", checkpoint, source_test_yaml)
    locked_external_eval(checkpoint, report_root, imgsz)


def aggregate() -> None:
    rows = []
    for seed in SEEDS:
        for stage in ("source_only", "model2_balanced", "stage1_ssod_teacher", "stage2_refined"):
            report_root = REPORT_ROOT / f"seed_{seed}" / stage
            internal = next(csv.DictReader((report_root / "scene_disjoint_test_metrics.csv").open(encoding="utf-8-sig")))
            external = json.loads((report_root / "locked_external" / "locked_external_protocol_summary.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "scene_map50": internal["map50"],
                    "scene_map5095": internal["map5095"],
                    "locked_external_map50": external["locked_final_detection_ap"]["map50"],
                    "locked_external_map5095": external["locked_final_detection_ap"]["map5095"],
                    "locked_external_precision": external["locked_final_global_threshold_metrics"]["macro_precision"],
                    "locked_external_recall": external["locked_final_global_threshold_metrics"]["macro_recall"],
                    "locked_external_f1": external["locked_final_global_threshold_metrics"]["macro_f1"],
                    "checkpoint": internal["checkpoint"],
                }
            )
    ensure_dir(REPORT_ROOT)
    with (REPORT_ROOT / "formal_seed_results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="source_only,model2_balanced,stage1_ssod_teacher,stage2_refined")
    parser.add_argument("--seeds", default="42,2026,3407")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    for stage in stages:
        for seed in seeds:
            print(f"\n=== protocol-v2 formal: stage={stage}, seed={seed} ===")
            run_one(stage, seed, args.epochs, args.imgsz, args.batch)
    if tuple(seeds) == SEEDS and set(stages) == {"source_only", "model2_balanced", "stage1_ssod_teacher", "stage2_refined"}:
        aggregate()


if __name__ == "__main__":
    main()

