"""Build reduced FP-hard and under_ripe-positive training subsets.

This script uses existing mined false-positive tables and labeled external
train/valid data. It does not read or copy the external test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    CLASS_NAME_TO_ID,
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    convert_to_yolo_format,
    ensure_dir,
    link_or_copy_file,
    list_images,
    load_label_boxes,
    set_safe_delete_root,
    write_data_yaml,
    write_rows_csv,
)


UNDER_RIPE_ID = CLASS_NAME_TO_ID["under_ripe"]


def read_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_filtered_label(src_label: Path, dst_label: Path, allowed_class_ids: Sequence[int]) -> int:
    ensure_dir(dst_label.parent)
    if not src_label.exists():
        dst_label.write_text("", encoding="utf-8")
        return 0
    allowed = {int(class_id) for class_id in allowed_class_ids}
    kept: List[str] = []
    for line in src_label.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id in allowed:
            kept.append(line)
    dst_label.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept)


def select_reduced_fp_cases(rows: Sequence[dict], per_class: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    by_class: Dict[str, List[dict]] = {}
    for row in rows:
        class_name = row.get("predicted_class", "")
        if class_name not in {"abnormal", "ripe", "under_ripe", "unripe"}:
            continue
        by_class.setdefault(class_name, []).append(row)

    selected: List[dict] = []
    for class_name, class_rows in sorted(by_class.items()):
        ordered = sorted(class_rows, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)
        high_conf = ordered[: max(per_class // 2, 0)]
        rest = ordered[max(per_class // 2, 0) :]
        rng.shuffle(rest)
        selected.extend((high_conf + rest)[:per_class])
    return selected


def build_fp_dataset(selected_rows: Sequence[dict], output_root: Path, allowed_class_ids: Sequence[int]) -> Path:
    dataset_root = output_root / "false_positive_hard_reduced"
    for split in ["train", "valid", "test"]:
        ensure_dir(dataset_root / split / "images")
        ensure_dir(dataset_root / split / "labels")
    for idx, row in enumerate(selected_rows):
        image_path = Path(row["image"])
        label_path = Path(row["label"])
        dst_stem = f"fpred_{idx:05d}"
        dst_image = dataset_root / "train" / "images" / f"{dst_stem}{image_path.suffix.lower()}"
        dst_label = dataset_root / "train" / "labels" / f"{dst_stem}.txt"
        link_or_copy_file(image_path, dst_image)
        write_filtered_label(label_path, dst_label, allowed_class_ids)
    write_data_yaml(dataset_root)
    return dataset_root


def collect_under_ripe_positive_rows(dataset_root: Path) -> List[dict]:
    rows: List[dict] = []
    for split in ["train", "valid"]:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in list_images(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = load_label_boxes(label_path, allowed_class_ids=set(PREPROCESSED_FFB_EVAL_CLASS_IDS))
            under_count = sum(1 for box in boxes if box.class_id == UNDER_RIPE_ID)
            if under_count:
                rows.append(
                    {
                        "image": str(image_path),
                        "label": str(label_path),
                        "split": split,
                        "under_ripe_boxes": under_count,
                        "total_boxes": len(boxes),
                    }
                )
    return rows


def select_under_ripe_positive_rows(rows: Sequence[dict], max_images: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda item: (int(item["under_ripe_boxes"]), int(item["total_boxes"])), reverse=True)
    high_density = ordered[: max(max_images // 2, 0)]
    rest = ordered[max(max_images // 2, 0) :]
    rng.shuffle(rest)
    return (high_density + rest)[:max_images]


def build_under_ripe_positive_dataset(selected_rows: Sequence[dict], output_root: Path, allowed_class_ids: Sequence[int]) -> Path:
    dataset_root = output_root / "under_ripe_positive_subset"
    for split in ["train", "valid", "test"]:
        ensure_dir(dataset_root / split / "images")
        ensure_dir(dataset_root / split / "labels")
    for idx, row in enumerate(selected_rows):
        image_path = Path(row["image"])
        label_path = Path(row["label"])
        dst_stem = f"urpos_{idx:05d}"
        dst_image = dataset_root / "train" / "images" / f"{dst_stem}{image_path.suffix.lower()}"
        dst_label = dataset_root / "train" / "labels" / f"{dst_stem}.txt"
        link_or_copy_file(image_path, dst_image)
        write_filtered_label(label_path, dst_label, allowed_class_ids)
    write_data_yaml(dataset_root)
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build precision-focused reduced training subsets.")
    parser.add_argument("--fp-cases-csv", type=Path, required=True)
    parser.add_argument("--external-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fp-per-class", type=int, default=250)
    parser.add_argument("--under-ripe-positive-max-images", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_safe_delete_root(args.output_root)
    ensure_dir(args.output_root)

    fp_rows = read_rows(args.fp_cases_csv)
    selected_fp_rows = select_reduced_fp_cases(fp_rows, per_class=args.fp_per_class, seed=args.seed)
    fp_dataset = build_fp_dataset(selected_fp_rows, args.output_root, PREPROCESSED_FFB_EVAL_CLASS_IDS)
    write_rows_csv(args.output_root / "selected_reduced_false_positive_cases.csv", selected_fp_rows)

    canonical_yaml = convert_to_yolo_format(args.external_dataset, args.output_root / "canonical_external")
    positive_rows = collect_under_ripe_positive_rows(canonical_yaml.parent)
    selected_positive_rows = select_under_ripe_positive_rows(
        positive_rows,
        max_images=args.under_ripe_positive_max_images,
        seed=args.seed,
    )
    positive_dataset = build_under_ripe_positive_dataset(
        selected_positive_rows,
        args.output_root,
        PREPROCESSED_FFB_EVAL_CLASS_IDS,
    )
    write_rows_csv(args.output_root / "selected_under_ripe_positive_cases.csv", selected_positive_rows)

    summary = {
        "fp_dataset": str(fp_dataset),
        "under_ripe_positive_dataset": str(positive_dataset),
        "test_split_used_for_training": False,
        "selected_fp_cases": len(selected_fp_rows),
        "selected_under_ripe_positive_images": len(selected_positive_rows),
        "fp_by_predicted_class": {},
    }
    for row in selected_fp_rows:
        class_name = row.get("predicted_class", "")
        summary["fp_by_predicted_class"][class_name] = summary["fp_by_predicted_class"].get(class_name, 0) + 1
    (args.output_root / "precision_training_subsets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

