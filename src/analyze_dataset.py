"""Generate strict dataset protocol reports without training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SCI-style split and duplicate reports.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--enable-phash-duplicate-check", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    protocol = cfg.get("protocol", {})
    scene = protocol.get("scene_disjoint", {})
    model = cfg.get("model", {})
    datasets = cfg.get("datasets", {})
    output_root = Path(cfg.get("project", {}).get("output_root", "paper_framework"))

    command = [
        sys.executable,
        str(Path(__file__).with_name("full_ssod_ffb_pipeline.py")),
        "--output-root",
        str(output_root),
        "--original-dataset",
        datasets["in_domain_sciencedb"],
        "--external-test-dataset",
        datasets["external_preprocessed_ffb"],
        "--new-images",
        datasets.get("unlabeled_new_images", ""),
        "--protocol-report-only",
        "--self-training-iterations",
        "0",
        "--max-epochs",
        str(model.get("max_epochs", 10)),
        "--imgsz",
        str(model.get("imgsz", 640)),
        "--batch",
        str(model.get("batch", 8)),
        "--device",
        str(model.get("device", "0")),
    ]
    if scene.get("val_groups"):
        command += ["--scene-val-groups", scene["val_groups"]]
    if scene.get("test_groups"):
        command += ["--scene-test-groups", scene["test_groups"]]
    if args.enable_phash_duplicate_check or protocol.get("duplicate_check", {}).get("perceptual_hash", False):
        command.append("--enable-phash-duplicate-check")

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

