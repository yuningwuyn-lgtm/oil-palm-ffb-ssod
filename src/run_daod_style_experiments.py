"""Run lightweight DAOD-style SSOD proxies on the oil-palm FFB pipeline.

These experiments are intentionally named "style" rather than exact paper
reproductions. The Ultralytics training API is kept intact, so this runner uses
offline teacher pseudo-labeling:

- at_weak_strong: Adaptive Teacher-inspired weak/strong consistency.
- mic_masked: adds MIC-inspired coordinate-preserving patch masking.

The external test labels are used only for evaluation. Target pseudo-labeling
uses the unlabeled new-images folder without folder-class filtering.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"LOCAL_PROJECT_ROOT")
PIPELINE = PROJECT_ROOT / "pythonProject1" / "full_ssod_ffb_pipeline.py"
SOURCE_PREPARED = PROJECT_ROOT / "paper_framework" / "datasets" / "prepared_supervised" / "local_original"
BASELINE_MODEL = PROJECT_ROOT / "paper_framework" / "weights" / "baseline_best.pt"
EXTERNAL_DATASET = Path(r"LOCAL_DATA_ROOT\preprocessed-ffb\yolov8")
UNLABELED_IMAGES = Path(r"LOCAL_DATA_ROOT\new_images_raw_ffb_augmented_for_ssod")


def command_for(profile: str, output_root: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE),
        "--output-root",
        str(output_root),
        "--prepared-supervised-root",
        str(SOURCE_PREPARED),
        "--external-test-dataset",
        str(EXTERNAL_DATASET),
        "--new-images",
        str(UNLABELED_IMAGES),
        "--skip-baseline-training",
        "--baseline-model-path",
        str(BASELINE_MODEL),
        "--teacher-model",
        str(BASELINE_MODEL),
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
        profile,
        "--num-aug-views",
        str(args.num_aug_views),
        "--min-consistency-support",
        str(args.min_consistency_support),
        "--allow-folder-mismatch",
    ]
    if profile == "mic_masked":
        command.extend(
            [
                "--prepared-external-masked-root",
                str(args.output_base / f"seed_{args.seed}" / "at_weak_strong" / "datasets" / "external_test_preprocessed_ffb_masked_4class"),
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AT-style and MIC-style FFB UDA proxy experiments.")
    parser.add_argument("--output-base", type=Path, default=Path(r"LOCAL_DATA_ROOT\paper_daod_style"))
    parser.add_argument("--profiles", default="at_weak_strong,mic_masked")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--train-cache", default="disk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-quota", type=int, default=100)
    parser.add_argument("--max-pseudo-images-per-class", type=int, default=100)
    parser.add_argument("--num-aug-views", type=int, default=4)
    parser.add_argument("--min-consistency-support", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for profile in [value.strip() for value in args.profiles.split(",") if value.strip()]:
        output_root = args.output_base / f"seed_{args.seed}" / profile
        command = command_for(profile, output_root, args)
        print("\nRUN:", subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

