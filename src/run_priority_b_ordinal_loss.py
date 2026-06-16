"""Run priority-B ordinal maturity auxiliary-loss ablation.

This reuses the retained final model's prepared supervised dataset so that the
only method change is the ordinal auxiliary classification loss.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "full_ssod_ffb_pipeline.py"
PREPARED_SUPERVISED = Path(
    r"LOCAL_DATA_ROOT\paper_stage2_formal\seed_42"
    r"\model2_ssod_fp250_urpos640_yolov8n\datasets\prepared_supervised_merged"
)
EXTERNAL_MASKED_CACHE = Path(
    r"LOCAL_DATA_ROOT\paper_daod_style\seed_42\at_weak_strong"
    r"\datasets\external_test_preprocessed_ffb_masked_4class"
)
UNLABELED = Path(r"LOCAL_DATA_ROOT\new_images_raw_ffb_augmented_for_ssod")
STAGE1_TEACHER = Path(
    r"LOCAL_DATA_ROOT\paper_stage1_fast\seed_42"
    r"\stage1_model2_balanced_ssod_hard_relabel\runs\self_train_iter_1\weights\best.pt"
)
DEFAULT_OUTPUT_ROOT = Path(r"LOCAL_DATA_ROOT\paper_priority_runs\priority_b_ordinal015_seed42")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ordinal maturity loss ablation.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--ordinal-loss-gain", type=float, default=0.15)
    args = parser.parse_args()

    for required in [PIPELINE, PREPARED_SUPERVISED / "data.yaml", EXTERNAL_MASKED_CACHE / "data.yaml", UNLABELED, STAGE1_TEACHER]:
        if not required.exists():
            raise FileNotFoundError(required)

    command = [
        sys.executable,
        str(PIPELINE),
        "--output-root",
        str(args.output_root),
        "--prepared-supervised-root",
        str(PREPARED_SUPERVISED),
        "--prepared-external-masked-root",
        str(EXTERNAL_MASKED_CACHE),
        "--new-images",
        str(UNLABELED),
        "--base-model",
        str(STAGE1_TEACHER),
        "--teacher-model",
        str(STAGE1_TEACHER),
        "--max-epochs",
        str(args.epochs),
        "--self-training-iterations",
        "0",
        "--imgsz",
        "640",
        "--batch",
        str(args.batch),
        "--device",
        "0",
        "--train-cache",
        "disk",
        "--seed",
        "42",
        "--ordinal-loss-gain",
        str(args.ordinal_loss_gain),
        "--target-operating-precision",
        "0.85",
        "--skip-protocol-reports",
    ]
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()


