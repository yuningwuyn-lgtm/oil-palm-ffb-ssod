"""Run priority-A three-class boundary sampling ablation.

The experiment keeps the retained final recipe fixed except that the
under_ripe-positive supplement is replaced by a compact, balanced
unripe-under_ripe-ripe boundary subset. External test and Outdoor Zenodo are
evaluation-only and are never added to training.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "full_ssod_ffb_pipeline.py"
BOUNDARY_BUILDER = HERE / "build_under_ripe_boundary_subset.py"

SOURCE = Path(r"LOCAL_DATA_ROOT\data_Annotated Video Dataset of Oil Palm Fruit Bunch Piles for Ripeness Grading")
EXTERNAL = Path(r"LOCAL_DATA_ROOT\preprocessed-ffb\yolov8")
UNLABELED = Path(r"LOCAL_DATA_ROOT\new_images_raw_ffb_augmented_for_ssod")
STAGE1_TEACHER = Path(r"LOCAL_DATA_ROOT\paper_stage1_fast\seed_42\stage1_model2_balanced_ssod_hard_relabel\runs\self_train_iter_1\weights\best.pt")
BALANCED_EXTERNAL = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\preprocessed_ffb_class_balanced\target_800\balanced")
FP_HARD_REDUCED = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\precision_target085_reduced_subsets\false_positive_hard_reduced")
BOUNDARY_ROOT = Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\boundary_sampling_250_seed42")
BOUNDARY_DATASET = BOUNDARY_ROOT / "under_ripe_boundary_subset"
EXTERNAL_MASKED_CACHE = Path(r"LOCAL_DATA_ROOT\paper_daod_style\seed_42\at_weak_strong\datasets\external_test_preprocessed_ffb_masked_4class")
DEFAULT_OUTPUT_ROOT = Path(r"LOCAL_DATA_ROOT\paper_priority_runs\priority_a_boundary250_seed42_stable")


def run(command: list[str]) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compact three-class boundary-sampling ablation.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    for required in [SOURCE, EXTERNAL, UNLABELED, STAGE1_TEACHER, BALANCED_EXTERNAL, FP_HARD_REDUCED, EXTERNAL_MASKED_CACHE]:
        if not required.exists():
            raise FileNotFoundError(required)

    if not (BOUNDARY_DATASET / "data.yaml").exists():
        run(
            [
                sys.executable,
                str(BOUNDARY_BUILDER),
                "--external-dataset",
                str(EXTERNAL),
                "--output-root",
                str(BOUNDARY_ROOT),
                "--max-under-ripe",
                "250",
                "--max-ripe-negatives",
                "250",
                "--max-unripe-negatives",
                "250",
                "--seed",
                "42",
            ]
        )

    run(
        [
            sys.executable,
            str(PIPELINE),
            "--output-root",
            str(args.output_root),
            "--original-dataset",
            str(SOURCE),
            "--extra-supervised-dataset",
            str(BALANCED_EXTERNAL),
            "--extra-supervised-dataset",
            str(BOUNDARY_DATASET),
            "--extra-supervised-dataset",
            str(FP_HARD_REDUCED),
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
            "--target-operating-precision",
            "0.85",
            "--scene-quota-per-iteration",
            "100",
            "--max-pseudo-images-per-class",
            "100",
            "--reuse-prepared-dataset",
            "--skip-protocol-reports",
        ]
    )


if __name__ == "__main__":
    main()

