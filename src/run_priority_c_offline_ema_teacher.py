"""Run priority-C offline iterative EMA teacher-student baseline.

Ultralytics writes each student checkpoint from its in-training EMA weights.
The next self-training iteration uses that EMA checkpoint as its teacher.  This
is an offline iterative EMA baseline, not a batch-wise Adaptive Teacher
reproduction.  Ensemble voting and ordinal rescue stay disabled so the effect
of teacher refresh remains isolated.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "full_ssod_ffb_pipeline.py"
MIXED_RUN = Path(r"LOCAL_DATA_ROOT\paper_stage2_formal\seed_42\model2_balanced640_yolov8n_no_ssod")
MIXED_DATASET = MIXED_RUN / "datasets" / "prepared_supervised_merged"
MIXED_TEACHER = MIXED_RUN / "runs" / "baseline_supervised" / "weights" / "best.pt"
EXTERNAL_MASKED = Path(
    r"LOCAL_DATA_ROOT\paper_daod_style\seed_42\at_weak_strong"
    r"\datasets\external_test_preprocessed_ffb_masked_4class"
)
UNLABELED_IMAGES = Path(r"LOCAL_DATA_ROOT\new_images_raw_ffb_augmented_for_ssod")
DEFAULT_OUTPUT_ROOT = Path(r"LOCAL_DATA_ROOT\paper_priority_runs\priority_c_offline_ema_teacher_seed42")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated offline iterative EMA teacher baseline.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    required = [PIPELINE, MIXED_DATASET / "data.yaml", MIXED_TEACHER, EXTERNAL_MASKED / "data.yaml", UNLABELED_IMAGES]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required experiment inputs: {missing}")

    command = [
        sys.executable,
        str(PIPELINE),
        "--output-root",
        str(args.output_root),
        "--prepared-supervised-root",
        str(MIXED_DATASET),
        "--prepared-external-masked-root",
        str(EXTERNAL_MASKED),
        "--new-images",
        str(UNLABELED_IMAGES),
        "--skip-baseline-training",
        "--baseline-model-path",
        str(MIXED_TEACHER),
        "--teacher-model",
        str(MIXED_TEACHER),
        "--skip-protocol-reports",
        "--max-epochs",
        str(args.epochs),
        "--self-training-iterations",
        str(args.iterations),
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
        "--scene-quota-per-iteration",
        "100",
        "--max-pseudo-images-per-class",
        "100",
        "--consistency-profile",
        "at_weak_strong",
        "--num-aug-views",
        "4",
        "--min-consistency-support",
        "2",
        "--allow-folder-mismatch",
        "--self-train-from-teacher",
    ]
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()


