"""Run the frozen precision-first SSOD candidate across three formal seeds.

The candidate is selected before this runner starts. Training and calibration
never read locked-final. Locked-final evaluation runs only after every seed has
finished, so it cannot influence candidate selection or hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from full_ssod_ffb_pipeline import ensure_dir, evaluate_model


SEEDS = (42, 2026, 3407)
SOURCE_ROOT = Path(
    r"LOCAL_PROJECT_ROOT\paper_framework\datasets\protocol_repair_v2"
    r"\science_oilpalm_scene_coverage_disjoint"
)
EXTERNAL_ROOT = Path(
    r"LOCAL_PROJECT_ROOT\paper_framework\datasets\protocol_repair_v2"
    r"\external_preprocessed_ffb_calibration_locked"
)
SUPERVISED_ROOT = Path(
    r"LOCAL_DATA_ROOT\paper_protocol_v2_formal\datasets"
    r"\model2_balanced"
)
NEW_IMAGES_ROOT = Path(
    r"LOCAL_DATA_ROOT\new_images_raw_ffb_augmented_for_ssod"
)
MODEL2_ROOT = Path(r"LOCAL_DATA_ROOT\paper_protocol_v2_formal")
OUTPUT_ROOT = Path(
    r"LOCAL_DATA_ROOT\paper_precision_first_ssod_formal"
)
REPORT_ROOT = Path(
    r"LOCAL_PROJECT_ROOT\paper_framework\reports"
    r"\precision_first_ssod_formal"
)


def trainer_args(imgsz: int) -> SimpleNamespace:
    return SimpleNamespace(imgsz=imgsz, device="0")


def model2_checkpoint(seed: int) -> Path:
    return (
        MODEL2_ROOT
        / f"seed_{seed}"
        / "model2_balanced"
        / "runs"
        / "formal_train"
        / "weights"
        / "best.pt"
    )


def strict_checkpoint(seed: int) -> Path:
    return (
        OUTPUT_ROOT
        / f"seed_{seed}"
        / "strict_ssod_no_relabel"
        / "runs"
        / "self_train_iter_1"
        / "weights"
        / "best.pt"
    )


def run_command(command: list[str]) -> None:
    print("\n>>>", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def train_seed(seed: int, epochs: int, imgsz: int, batch: int) -> Path:
    """Train the frozen B candidate without any locked-final access."""
    checkpoint = strict_checkpoint(seed)
    if checkpoint.exists():
        print(f"Reusing completed checkpoint: {checkpoint}", flush=True)
        return checkpoint
    teacher = model2_checkpoint(seed)
    if not teacher.exists():
        raise FileNotFoundError(f"Missing Model2 Balanced teacher: {teacher}")
    output_root = OUTPUT_ROOT / f"seed_{seed}" / "strict_ssod_no_relabel"
    command = [
        sys.executable,
        str(Path(__file__).with_name("full_ssod_ffb_pipeline.py")),
        "--prepared-supervised-root", str(SUPERVISED_ROOT),
        "--new-images", str(NEW_IMAGES_ROOT),
        "--output-root", str(output_root),
        "--skip-baseline-training",
        "--baseline-model-path", str(teacher),
        "--teacher-model", str(teacher),
        "--skip-protocol-reports",
        "--self-training-iterations", "1",
        "--max-epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--device", "0",
        "--train-cache", "disk",
        "--seed", str(seed),
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
    run_command(command)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training did not create checkpoint: {checkpoint}")
    return checkpoint


def write_source_metrics(seed: int, checkpoint: Path, imgsz: int) -> None:
    """Evaluate the source scene-disjoint test after training."""
    output_root = REPORT_ROOT / f"seed_{seed}"
    ensure_dir(output_root)
    output_csv = output_root / "scene_disjoint_test_metrics.csv"
    if output_csv.exists():
        return
    metrics = evaluate_model(
        checkpoint,
        SOURCE_ROOT / "data.yaml",
        "test",
        trainer_args(imgsz),
    )
    row = {
        "seed": seed,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map5095": float(metrics.box.map),
        "checkpoint": str(checkpoint),
    }
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_calibration_metrics(seed: int, checkpoint: Path, imgsz: int) -> None:
    """Fit deployment thresholds on calibration only."""
    output_root = REPORT_ROOT / f"seed_{seed}" / "calibration_only"
    summary = output_root / "calibration_only_summary.json"
    if summary.exists():
        return
    run_command(
        [
            sys.executable,
            str(Path(__file__).with_name("evaluate_calibration_only_protocol.py")),
            "--model", str(checkpoint),
            "--dataset-root", str(EXTERNAL_ROOT),
            "--output-root", str(output_root),
            "--imgsz", str(imgsz),
        ]
    )


def write_locked_final_metrics(seed: int, checkpoint: Path, imgsz: int) -> None:
    """Evaluate locked-final after candidate freezing and all training runs."""
    output_root = REPORT_ROOT / f"seed_{seed}" / "locked_external"
    summary = output_root / "locked_external_protocol_summary.json"
    if summary.exists():
        return
    run_command(
        [
            sys.executable,
            str(Path(__file__).with_name("evaluate_locked_external_protocol.py")),
            "--model", str(checkpoint),
            "--dataset-root", str(EXTERNAL_ROOT),
            "--output-root", str(output_root),
            "--imgsz", str(imgsz),
        ]
    )


def aggregate(seeds: list[int], include_locked_final: bool) -> None:
    rows = []
    for seed in seeds:
        root = REPORT_ROOT / f"seed_{seed}"
        source = next(
            csv.DictReader(
                (root / "scene_disjoint_test_metrics.csv").open(encoding="utf-8-sig")
            )
        )
        calibration = json.loads(
            (root / "calibration_only" / "calibration_only_summary.json").read_text(
                encoding="utf-8"
            )
        )
        row = {
            "seed": seed,
            "scene_map50": source["map50"],
            "scene_map5095": source["map5095"],
            "calibration_map50": calibration["calibration_detection_ap"]["map50"],
            "calibration_map5095": calibration["calibration_detection_ap"]["map5095"],
            "calibration_precision": calibration["calibration_classwise_metrics"][
                "macro_precision"
            ],
            "calibration_recall": calibration["calibration_classwise_metrics"][
                "macro_recall"
            ],
            "calibration_f1": calibration["calibration_classwise_metrics"]["macro_f1"],
            "checkpoint": source["checkpoint"],
        }
        if include_locked_final:
            locked = json.loads(
                (
                    root
                    / "locked_external"
                    / "locked_external_protocol_summary.json"
                ).read_text(encoding="utf-8")
            )
            row.update(
                {
                    "locked_external_map50": locked["locked_final_detection_ap"]["map50"],
                    "locked_external_map5095": locked["locked_final_detection_ap"][
                        "map5095"
                    ],
                    "locked_external_precision": locked[
                        "locked_final_global_threshold_metrics"
                    ]["macro_precision"],
                    "locked_external_recall": locked[
                        "locked_final_global_threshold_metrics"
                    ]["macro_recall"],
                    "locked_external_f1": locked[
                        "locked_final_global_threshold_metrics"
                    ]["macro_f1"],
                }
            )
        rows.append(row)
    ensure_dir(REPORT_ROOT)
    suffix = "final" if include_locked_final else "prelocked"
    output_csv = REPORT_ROOT / f"strict_ssod_formal_{suffix}_seed_results.csv"
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved aggregate: {output_csv}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,2026,3407")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--skip-locked-final", action="store_true")
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(REPORT_ROOT)
    checkpoints = {}
    for seed in seeds:
        print(f"\n=== frozen precision-first SSOD training: seed={seed} ===", flush=True)
        checkpoint = train_seed(seed, args.epochs, args.imgsz, args.batch)
        checkpoints[seed] = checkpoint
        write_source_metrics(seed, checkpoint, args.imgsz)
        write_calibration_metrics(seed, checkpoint, args.imgsz)
    aggregate(seeds, include_locked_final=False)
    if args.skip_locked_final:
        print("Locked-final evaluation skipped by request.", flush=True)
        return
    marker = REPORT_ROOT / "FROZEN_CANDIDATE.txt"
    marker.write_text(
        "B_strict_no_relabel frozen before locked-final evaluation.\n"
        f"seeds={','.join(str(seed) for seed in seeds)}\n"
        f"epochs={args.epochs}\nimgsz={args.imgsz}\nbatch={args.batch}\n",
        encoding="ascii",
    )
    for seed in seeds:
        print(f"\n=== locked-final evaluation after freeze: seed={seed} ===", flush=True)
        write_locked_final_metrics(seed, checkpoints[seed], args.imgsz)
    aggregate(seeds, include_locked_final=True)


if __name__ == "__main__":
    main()

