"""Evaluate a detector on preprocessed-ffb with the strict 4-class mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from full_ssod_ffb_pipeline import (
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    JsonRunLogger,
    convert_to_yolo_format,
    create_masked_external_dataset,
    ensure_dir,
    evaluate_model_masked_classes,
    report_path,
    set_safe_delete_root,
)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run masked external-domain evaluation only.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--model", type=Path, default=None, help="Model path. Defaults to config model.baseline_model_path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["project"]["output_root"])
    ensure_dir(output_root)
    set_safe_delete_root(output_root)
    logger = JsonRunLogger(output_root / "reports" / "structured_results_external_eval.json")

    external = Path(cfg["datasets"]["external_preprocessed_ffb"])
    model_path = args.model or Path(cfg["model"]["baseline_model_path"])
    canonical_yaml = convert_to_yolo_format(external, output_root / "datasets" / "external_eval_canonical")
    masked_yaml = create_masked_external_dataset(
        canonical_yaml.parent,
        output_root / "datasets" / "external_eval_preprocessed_ffb_masked_4class",
        PREPROCESSED_FFB_EVAL_CLASS_IDS,
    )

    ns = argparse.Namespace(
        imgsz=cfg["model"].get("imgsz", 640),
        device=str(cfg["model"].get("device", "0")),
        paper_layout=True,
    )
    summary = evaluate_model_masked_classes(
        model_path,
        masked_yaml.parent,
        split="test",
        allowed_class_ids=PREPROCESSED_FFB_EVAL_CLASS_IDS,
        args=ns,
        output_csv=report_path(ns, output_root, "external_preprocessed_ffb_metrics.csv"),
    )
    logger.log("external_masked_evaluation", model=model_path, data_yaml=masked_yaml, summary=summary)


if __name__ == "__main__":
    main()

