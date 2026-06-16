"""Run the strong-teacher balanced SSOD experiment for the paper pipeline.

The primary teacher is the fair mixed-supervised YOLOv8n baseline. Additional
adapted models vote on pseudo labels but do not replace the primary teacher.
The ordinal rescue is restricted to adjacent maturity classes inside weakly
labeled under_ripe folders.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"LOCAL_PROJECT_ROOT")
PIPELINE = PROJECT_ROOT / "pythonProject1" / "full_ssod_ffb_pipeline.py"
MIXED_RUN = Path(r"LOCAL_DATA_ROOT\paper_stage2_formal\seed_42\model2_balanced640_yolov8n_no_ssod")
MIXED_DATASET = MIXED_RUN / "datasets" / "prepared_supervised_merged"
MIXED_TEACHER = MIXED_RUN / "runs" / "baseline_supervised" / "weights" / "best.pt"
VOTER_YOLOV8N = Path(r"LOCAL_DATA_ROOT\paper_stage2_formal\seed_42\model2_ssod_fp250_urpos640_yolov8n\runs\baseline_supervised\weights\best.pt")
VOTER_YOLOV8S = Path(r"LOCAL_DATA_ROOT\paper_stage2_formal\seed_42\model2_fp250_urpos640_yolov8s\runs\baseline_supervised\weights\best.pt")
EXTERNAL_MASKED = Path(r"LOCAL_DATA_ROOT\paper_daod_style\seed_42\at_weak_strong\datasets\external_test_preprocessed_ffb_masked_4class")
EXTERNAL_RAW = Path(r"LOCAL_DATA_ROOT\preprocessed-ffb\yolov8")
UNLABELED_IMAGES = Path(r"LOCAL_DATA_ROOT\new_images_raw_ffb_augmented_for_ssod")


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(PIPELINE),
        "--output-root",
        str(args.output_root),
        "--prepared-supervised-root",
        str(MIXED_DATASET),
        "--external-test-dataset",
        str(EXTERNAL_RAW),
        "--prepared-external-masked-root",
        str(EXTERNAL_MASKED),
        "--new-images",
        str(UNLABELED_IMAGES),
        "--skip-baseline-training",
        "--baseline-model-path",
        str(MIXED_TEACHER),
        "--teacher-model",
        str(MIXED_TEACHER),
        "--ensemble-teacher-model",
        str(VOTER_YOLOV8N),
        "--ensemble-teacher-model",
        str(VOTER_YOLOV8S),
        "--min-ensemble-support",
        "2",
        "--ensemble-iou",
        "0.55",
        "--enable-ordinal-under-ripe-rescue",
        "--ordinal-min-quality",
        "0.72",
        "--ordinal-min-conf",
        "0.45",
        "--ordinal-min-view-support",
        "2",
        "--ordinal-min-ensemble-support",
        "2",
        "--skip-protocol-reports",
        "--max-epochs",
        str(args.epochs),
        "--self-training-iterations",
        "1",
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--train-cache",
        args.train_cache,
        "--seed",
        str(args.seed),
        "--scene-quota-per-iteration",
        str(args.scene_quota),
        "--max-pseudo-images-per-class",
        str(args.max_pseudo_images_per_class),
        "--consistency-profile",
        "at_weak_strong",
        "--num-aug-views",
        "4",
        "--min-consistency-support",
        "2",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mixed-teacher ensemble ordinal SSOD.")
    parser.add_argument("--output-root", type=Path, default=Path(r"LOCAL_DATA_ROOT\paper_stage3_ensemble_ordinal\seed_42"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--train-cache", default="disk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-quota", type=int, default=100)
    parser.add_argument("--max-pseudo-images-per-class", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    required = [PIPELINE, MIXED_DATASET / "data.yaml", MIXED_TEACHER, VOTER_YOLOV8N, VOTER_YOLOV8S, EXTERNAL_MASKED / "data.yaml", UNLABELED_IMAGES]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required experiment inputs: {missing}")

    command = build_command(args)
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

