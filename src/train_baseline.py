"""Train or evaluate the supervised in-domain YOLOv8 baseline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised baseline under the strict project layout.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--skip-training", action="store_true", help="Use configured baseline_model_path instead of training.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg["datasets"]
    model = cfg["model"]
    self_training = cfg.get("self_training", {})
    output_root = cfg["project"]["output_root"]

    command = [
        sys.executable,
        str(Path(__file__).with_name("full_ssod_ffb_pipeline.py")),
        "--output-root",
        output_root,
        "--original-dataset",
        datasets["in_domain_sciencedb"],
        "--external-test-dataset",
        datasets["external_preprocessed_ffb"],
        "--new-images",
        datasets.get("unlabeled_new_images", ""),
        "--base-model",
        model.get("base_model", "yolov8n.pt"),
        "--teacher-model",
        model.get("teacher_model", ""),
        "--max-epochs",
        str(model.get("max_epochs", 10)),
        "--self-training-iterations",
        "0",
        "--imgsz",
        str(model.get("imgsz", 640)),
        "--batch",
        str(model.get("batch", 8)),
        "--device",
        str(model.get("device", "0")),
        "--max-pseudo-images-per-class",
        str(self_training.get("max_pseudo_images_per_class", 0)),
    ]
    if args.skip_training:
        command += ["--skip-baseline-training", "--baseline-model-path", model["baseline_model_path"]]

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

