"""Resume priority-B ordinal training and regenerate its evaluation reports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

from ordinal_detection_trainer import make_ordinal_trainer


HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "full_ssod_ffb_pipeline.py"
OUTPUT_ROOT = Path(r"LOCAL_DATA_ROOT\paper_priority_runs\priority_b_ordinal015_seed42")
PREPARED_SUPERVISED = Path(
    r"LOCAL_DATA_ROOT\paper_stage2_formal\seed_42"
    r"\model2_ssod_fp250_urpos640_yolov8n\datasets\prepared_supervised_merged"
)
EXTERNAL_MASKED = Path(
    r"LOCAL_DATA_ROOT\paper_daod_style\seed_42\at_weak_strong"
    r"\datasets\external_test_preprocessed_ffb_masked_4class"
)
UNLABELED = Path(r"LOCAL_DATA_ROOT\new_images_raw_ffb_augmented_for_ssod")
TEACHER = Path(
    r"LOCAL_DATA_ROOT\paper_stage1_fast\seed_42"
    r"\stage1_model2_balanced_ssod_hard_relabel\runs\self_train_iter_1\weights\best.pt"
)
LAST = OUTPUT_ROOT / "runs" / "baseline_supervised" / "weights" / "last.pt"
BEST = OUTPUT_ROOT / "runs" / "baseline_supervised" / "weights" / "best.pt"


def main() -> None:
    for required in [LAST, PREPARED_SUPERVISED / "data.yaml", EXTERNAL_MASKED / "data.yaml", UNLABELED, TEACHER]:
        if not required.exists():
            raise FileNotFoundError(required)

    print(f"Resuming ordinal training from: {LAST}", flush=True)
    model = YOLO(str(LAST))
    model.train(trainer=make_ordinal_trainer(0.15), resume=str(LAST), workers=0)

    command = [
        sys.executable,
        str(PIPELINE),
        "--output-root",
        str(OUTPUT_ROOT),
        "--prepared-supervised-root",
        str(PREPARED_SUPERVISED),
        "--prepared-external-masked-root",
        str(EXTERNAL_MASKED),
        "--new-images",
        str(UNLABELED),
        "--skip-baseline-training",
        "--baseline-model-path",
        str(BEST),
        "--teacher-model",
        str(TEACHER),
        "--max-epochs",
        "10",
        "--self-training-iterations",
        "0",
        "--imgsz",
        "640",
        "--batch",
        "2",
        "--device",
        "0",
        "--seed",
        "42",
        "--target-operating-precision",
        "0.85",
        "--skip-protocol-reports",
    ]
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()


