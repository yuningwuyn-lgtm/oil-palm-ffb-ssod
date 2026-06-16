"""Build an under_ripe boundary-focused supervised subset.

The subset contains:
- under_ripe-positive images, emphasizing samples with more under_ripe boxes
- ripe-only boundary negatives
- unripe-only boundary negatives

Only external train/valid splits are used. External test is never copied.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from full_ssod_ffb_pipeline import (
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


UNDER = CLASS_NAME_TO_ID["under_ripe"]
RIPE = CLASS_NAME_TO_ID["ripe"]
UNRIPE = CLASS_NAME_TO_ID["unripe"]


def write_filtered_label(src_label: Path, dst_label: Path, allowed_class_ids: Sequence[int]) -> int:
    ensure_dir(dst_label.parent)
    allowed = set(allowed_class_ids)
    kept: List[str] = []
    if src_label.exists():
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


def collect_rows(dataset_root: Path) -> List[dict]:
    rows: List[dict] = []
    for split in ["train", "valid"]:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in list_images(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = load_label_boxes(label_path, allowed_class_ids=set(PREPROCESSED_FFB_EVAL_CLASS_IDS))
            counts: Dict[int, int] = {}
            for box in boxes:
                counts[box.class_id] = counts.get(box.class_id, 0) + 1
            if counts.get(UNDER, 0):
                group = "under_ripe_positive"
            elif counts.get(RIPE, 0):
                group = "ripe_boundary_negative"
            elif counts.get(UNRIPE, 0):
                group = "unripe_boundary_negative"
            else:
                continue
            rows.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "split": split,
                    "group": group,
                    "under_ripe_boxes": counts.get(UNDER, 0),
                    "ripe_boxes": counts.get(RIPE, 0),
                    "unripe_boxes": counts.get(UNRIPE, 0),
                    "total_boxes": sum(counts.values()),
                }
            )
    return rows


def select_rows(rows: Sequence[dict], max_under: int, max_ripe_neg: int, max_unripe_neg: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    selected: List[dict] = []
    groups = {
        "under_ripe_positive": max_under,
        "ripe_boundary_negative": max_ripe_neg,
        "unripe_boundary_negative": max_unripe_neg,
    }
    for group, cap in groups.items():
        group_rows = [row for row in rows if row["group"] == group]
        if group == "under_ripe_positive":
            ordered = sorted(group_rows, key=lambda item: (int(item["under_ripe_boxes"]), int(item["total_boxes"])), reverse=True)
            front = ordered[: max(cap // 2, 0)]
            rest = ordered[max(cap // 2, 0) :]
        else:
            ordered = sorted(group_rows, key=lambda item: int(item["total_boxes"]), reverse=True)
            front = ordered[: max(cap // 3, 0)]
            rest = ordered[max(cap // 3, 0) :]
        rng.shuffle(rest)
        selected.extend((front + rest)[:cap])
    return selected


def build_dataset(selected_rows: Sequence[dict], output_root: Path, allowed_class_ids: Sequence[int]) -> Path:
    dataset_root = output_root / "under_ripe_boundary_subset"
    for split in ["train", "valid", "test"]:
        ensure_dir(dataset_root / split / "images")
        ensure_dir(dataset_root / split / "labels")
    for idx, row in enumerate(selected_rows):
        image_path = Path(row["image"])
        label_path = Path(row["label"])
        stem = f"urbound_{idx:05d}"
        dst_image = dataset_root / "train" / "images" / f"{stem}{image_path.suffix.lower()}"
        dst_label = dataset_root / "train" / "labels" / f"{stem}.txt"
        link_or_copy_file(image_path, dst_image)
        write_filtered_label(label_path, dst_label, allowed_class_ids)
    write_data_yaml(dataset_root)
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an under_ripe boundary-focused training subset.")
    parser.add_argument("--external-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-under-ripe", type=int, default=1000)
    parser.add_argument("--max-ripe-negatives", type=int, default=400)
    parser.add_argument("--max-unripe-negatives", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_safe_delete_root(args.output_root)
    ensure_dir(args.output_root)
    canonical_yaml = convert_to_yolo_format(args.external_dataset, args.output_root / "canonical_external")
    rows = collect_rows(canonical_yaml.parent)
    selected = select_rows(rows, args.max_under_ripe, args.max_ripe_negatives, args.max_unripe_negatives, args.seed)
    dataset_root = build_dataset(selected, args.output_root, PREPROCESSED_FFB_EVAL_CLASS_IDS)
    write_rows_csv(args.output_root / "selected_under_ripe_boundary_cases.csv", selected)

    summary = {
        "dataset": str(dataset_root),
        "data_yaml": str(dataset_root / "data.yaml"),
        "test_split_used_for_training": False,
        "selected_total": len(selected),
        "by_group": {},
    }
    for row in selected:
        summary["by_group"][row["group"]] = summary["by_group"].get(row["group"], 0) + 1
    (args.output_root / "under_ripe_boundary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

