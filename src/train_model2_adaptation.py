"""Run Model 2: supervised cross-domain adaptation.

This experiment trains from the ScienceDB baseline and adds the labeled
preprocessed-ffb train/valid data as external-domain adaptation data. The
external test remains evaluated with the 4-class mask.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-domain Model 2 adaptation.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Default is config model.model2_adaptation_epochs.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--base-model", type=str, default=None, help="Ultralytics init model or best.pt. Overrides config model.baseline_model_path.")
    parser.add_argument("--teacher-model", type=str, default=None, help="Teacher best.pt used for pseudo-labeling when SSOD is enabled.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--train-cache", type=str, default="none", help="Forwarded to Ultralytics cache: none, ram, or disk.")
    parser.add_argument("--scene-quota-per-iteration", type=int, default=None)
    parser.add_argument("--max-pseudo-images-per-class", type=int, default=None)
    parser.add_argument("--skip-protocol-reports", action="store_true")
    parser.add_argument("--reuse-prepared-dataset", action="store_true")
    parser.add_argument("--prepared-supervised-root", type=Path, default=None)
    parser.add_argument("--run-ssod-after-adaptation", action="store_true")
    parser.add_argument("--class-balanced-external", action="store_true", help="Use a pre-generated class-balanced preprocessed-ffb dataset as the external supervised source.")
    parser.add_argument("--balanced-external-root", type=Path, default=Path("LOCAL_PROJECT_ROOT/paper_framework/datasets/preprocessed_ffb_class_balanced"))
    parser.add_argument("--balanced-target-images-per-class", type=int, default=800, help="Cap/target external class-balanced train images per class. Use 0 for legacy max-class oversampling.")
    parser.add_argument("--hard-under-ripe-dataset", type=Path, default=None, help="Optional mined under_ripe hard-negative YOLO dataset added as an extra supervised source.")
    parser.add_argument("--hard-fp-dataset", type=Path, default=None, help="Optional all-class false-positive hard dataset added as an extra supervised source.")
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--quality-threshold-by-class", type=str, default="")
    parser.add_argument("--base-conf-threshold", type=float, default=None)
    parser.add_argument("--class-threshold-overrides", type=str, default="")
    parser.add_argument("--disable-consistency", action="store_true")
    parser.add_argument("--disable-quality-scoring", action="store_true")
    parser.add_argument("--disable-dynamic-thresholds", action="store_true")
    parser.add_argument("--enable-source-aware-loss-mask", action="store_true")
    parser.add_argument("--allow-folder-mismatch", action="store_true")
    parser.add_argument("--enable-folder-guided-hard-class-relabel", action="store_true")
    parser.add_argument("--hard-classes", type=str, default="")
    parser.add_argument("--hard-class-source-map", type=str, default="")
    parser.add_argument("--hard-class-min-quality", type=float, default=None)
    parser.add_argument("--hard-class-min-conf", type=float, default=None)
    parser.add_argument("--hard-class-min-support", type=int, default=None)
    parser.add_argument("--target-operating-precision", type=float, default=None, help="Forwarded external class-wise operating precision target, e.g. 0.85.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg["datasets"]
    model = cfg["model"]
    self_training = cfg.get("self_training", {})
    epochs = args.epochs or int(model.get("model2_adaptation_epochs", 3))
    output_root = args.output_root or Path("LOCAL_PROJECT_ROOT/paper_framework_model2_adapt_3e")

    external_dataset = Path(datasets["external_preprocessed_ffb"])
    if args.class_balanced_external:
        from full_ssod_ffb_pipeline import PREPROCESSED_FFB_EVAL_CLASS_IDS, convert_to_yolo_format, create_class_balanced_yolo_dataset, set_safe_delete_root

        balance_root = args.balanced_external_root / f"target_{args.balanced_target_images_per_class}"
        balance_root.mkdir(parents=True, exist_ok=True)
        set_safe_delete_root(balance_root)
        balanced_dataset_root = balance_root / "balanced"
        balanced_yaml_path = balanced_dataset_root / "data.yaml"
        if args.reuse_prepared_dataset and balanced_yaml_path.exists():
            balanced_yaml = balanced_yaml_path
        else:
            canonical_yaml = convert_to_yolo_format(external_dataset, balance_root / "canonical")
            balanced_yaml = create_class_balanced_yolo_dataset(
                canonical_yaml.parent,
                balanced_dataset_root,
                split="train",
                target_images_per_class=args.balanced_target_images_per_class,
                allowed_class_ids=PREPROCESSED_FFB_EVAL_CLASS_IDS,
                seed=42,
            )
        external_dataset = balanced_yaml.parent

    base_model = args.base_model or model.get("baseline_model_path", model.get("base_model", "yolov8n.pt"))
    teacher_model = args.teacher_model if args.teacher_model is not None else model.get("baseline_model_path", "")

    command = [
        sys.executable,
        str(Path(__file__).with_name("full_ssod_ffb_pipeline.py")),
        "--output-root",
        str(output_root),
        "--original-dataset",
        datasets["in_domain_sciencedb"],
        "--extra-supervised-dataset",
        str(external_dataset),
        "--external-test-dataset",
        datasets["external_preprocessed_ffb"],
        "--new-images",
        datasets.get("unlabeled_new_images", ""),
        "--base-model",
        base_model,
        "--teacher-model",
        teacher_model,
        "--max-epochs",
        str(epochs),
        "--self-training-iterations",
        str(self_training.get("iterations", 1) if args.run_ssod_after_adaptation else 0),
        "--imgsz",
        str(args.imgsz or model.get("imgsz", 640)),
        "--batch",
        str(args.batch or model.get("batch", 8)),
        "--device",
        str(model.get("device", "0")),
        "--train-cache",
        args.train_cache,
        "--seed",
        str(args.seed),
        "--scene-quota-per-iteration",
        str(args.scene_quota_per_iteration if args.scene_quota_per_iteration is not None else self_training.get("scene_quota_per_iteration", 250)),
        "--max-pseudo-images-per-class",
        str(args.max_pseudo_images_per_class if args.max_pseudo_images_per_class is not None else self_training.get("max_pseudo_images_per_class", 200)),
    ]
    if args.hard_under_ripe_dataset:
        command.extend(["--extra-supervised-dataset", str(args.hard_under_ripe_dataset)])
    if args.hard_fp_dataset:
        command.extend(["--extra-supervised-dataset", str(args.hard_fp_dataset)])
    if args.skip_protocol_reports:
        command.append("--skip-protocol-reports")
    if args.reuse_prepared_dataset:
        command.append("--reuse-prepared-dataset")
    if args.prepared_supervised_root:
        command.extend(["--prepared-supervised-root", str(args.prepared_supervised_root)])
    if args.quality_threshold is not None:
        command.extend(["--quality-threshold", str(args.quality_threshold)])
    if args.quality_threshold_by_class:
        command.extend(["--quality-threshold-by-class", args.quality_threshold_by_class])
    if args.base_conf_threshold is not None:
        command.extend(["--base-conf-threshold", str(args.base_conf_threshold)])
    if args.class_threshold_overrides:
        command.extend(["--class-threshold-overrides", args.class_threshold_overrides])
    if args.disable_consistency:
        command.append("--disable-consistency")
    if args.disable_quality_scoring:
        command.append("--disable-quality-scoring")
    if args.disable_dynamic_thresholds:
        command.append("--disable-dynamic-thresholds")
    if args.enable_source_aware_loss_mask:
        command.append("--enable-source-aware-loss-mask")
    if args.allow_folder_mismatch:
        command.append("--allow-folder-mismatch")
    if args.enable_folder_guided_hard_class_relabel:
        command.append("--enable-folder-guided-hard-class-relabel")
    if args.hard_classes:
        command.extend(["--hard-classes", args.hard_classes])
    if args.hard_class_source_map:
        command.extend(["--hard-class-source-map", args.hard_class_source_map])
    if args.hard_class_min_quality is not None:
        command.extend(["--hard-class-min-quality", str(args.hard_class_min_quality)])
    if args.hard_class_min_conf is not None:
        command.extend(["--hard-class-min-conf", str(args.hard_class_min_conf)])
    if args.hard_class_min_support is not None:
        command.extend(["--hard-class-min-support", str(args.hard_class_min_support)])
    if args.target_operating_precision is not None:
        command.extend(["--target-operating-precision", str(args.target_operating_precision)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

