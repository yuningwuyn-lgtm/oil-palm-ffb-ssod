"""Repair paper-grade evaluation splits without modifying historical artifacts."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from full_ssod_ffb_pipeline import (
    CANONICAL_CLASSES,
    PREPROCESSED_FFB_EVAL_CLASS_IDS,
    count_label_file,
    ensure_dir,
    list_images,
    link_or_copy_file,
    normalized_base_filename,
    source_group_from_image_name,
    write_data_yaml,
    write_rows_csv,
)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_record(image: Path, label: Path, destination: Path, split: str) -> None:
    ensure_dir(destination / split / "images")
    ensure_dir(destination / split / "labels")
    link_or_copy_file(image, destination / split / "images" / image.name)
    lines = label.read_text(encoding="utf-8", errors="ignore").splitlines()
    unique_lines = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
    (destination / split / "labels" / label.name).write_text(
        "\n".join(unique_lines) + ("\n" if unique_lines else ""),
        encoding="utf-8",
    )


def image_classes(label: Path, allowed: Iterable[int]) -> set[int]:
    _, _, counts = count_label_file(label)
    allowed_set = set(allowed)
    return {class_id for class_id, count in counts.items() if count > 0 and class_id in allowed_set}


def split_external_calibration_locked(
    source_root: Path,
    output_root: Path,
    calibration_fraction: float,
    seed: int,
) -> dict:
    """Split the historical target test into grouped calibration and locked-final views."""
    clear_dir(output_root)
    for split in ("train", "valid", "calibration", "locked_final_test", "test"):
        ensure_dir(output_root / split / "images")
        ensure_dir(output_root / split / "labels")

    # Preserve train/valid only as provenance. They remain adaptation sources.
    for split in ("train", "valid"):
        for image in list_images(source_root / split / "images"):
            copy_record(image, source_root / split / "labels" / f"{image.stem}.txt", output_root, split)

    groups: dict[str, list[tuple[Path, Path, set[int]]]] = defaultdict(list)
    for image in list_images(source_root / "test" / "images"):
        label = source_root / "test" / "labels" / f"{image.stem}.txt"
        groups[normalized_base_filename(image)].append(
            (image, label, image_classes(label, PREPROCESSED_FFB_EVAL_CLASS_IDS))
        )

    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    class_totals = Counter()
    for key in keys:
        for _, _, classes in groups[key]:
            class_totals.update(classes)
    calibration_target = {class_id: max(1, round(count * calibration_fraction)) for class_id, count in class_totals.items()}

    calibration_keys: set[str] = set()
    calibration_counts = Counter()
    # Greedy class-aware allocation keeps Roboflow variants with the same base frame together.
    for key in sorted(keys, key=lambda item: (len({c for _, _, cs in groups[item] for c in cs}), len(groups[item])), reverse=True):
        key_classes = Counter()
        for _, _, classes in groups[key]:
            key_classes.update(classes)
        useful = any(calibration_counts[class_id] < calibration_target[class_id] for class_id in key_classes)
        if useful and len(calibration_keys) < max(1, round(len(keys) * calibration_fraction)):
            calibration_keys.add(key)
            calibration_counts.update(key_classes)
    for key in keys:
        if len(calibration_keys) >= max(1, round(len(keys) * calibration_fraction)):
            break
        calibration_keys.add(key)

    rows = []
    locked_keys = set(keys) - calibration_keys
    for key, records in groups.items():
        split = "calibration" if key in calibration_keys else "locked_final_test"
        for image, label, classes in records:
            copy_record(image, label, output_root, split)
            if split == "locked_final_test":
                copy_record(image, label, output_root, "test")
            rows.append(
                {
                    "normalized_base": key,
                    "split": split,
                    "image": str(image),
                    "classes": ";".join(CANONICAL_CLASSES[class_id] for class_id in sorted(classes)),
                }
            )
    write_rows_csv(output_root / "external_locked_split_manifest.csv", rows)
    write_data_yaml(output_root)
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "historical_note": "The source test was previously inspected. This repair creates a prospectively locked final subset for all future experiments.",
        "calibration_groups": len(calibration_keys),
        "locked_final_groups": len(locked_keys),
        "calibration_images": sum(1 for row in rows if row["split"] == "calibration"),
        "locked_final_images": sum(1 for row in rows if row["split"] == "locked_final_test"),
        "normalized_group_overlap": len(calibration_keys & locked_keys),
    }
    return summary


def collect_scene_records(source_root: Path) -> dict[str, list[tuple[Path, Path, set[int]]]]:
    groups: dict[str, list[tuple[Path, Path, set[int]]]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        for image in list_images(source_root / split / "images"):
            label = source_root / split / "labels" / f"{image.stem}.txt"
            groups[source_group_from_image_name(image.name)].append(
                (image, label, image_classes(label, CANONICAL_CLASSES))
            )
    return groups


def class_counts(records: Sequence[tuple[Path, Path, set[int]]]) -> Counter:
    counts = Counter()
    for _, label, _ in records:
        _, _, box_counts = count_label_file(label)
        counts.update({class_id: count for class_id, count in box_counts.items() if count})
    return counts


def choose_coverage_groups(groups: dict[str, list[tuple[Path, Path, set[int]]]]) -> tuple[set[str], set[str]]:
    """Choose non-overlapping val/test groups maximizing class coverage and balance."""
    names = sorted(groups)
    all_classes = set(CANONICAL_CLASSES)
    group_counts = {name: class_counts(records) for name, records in groups.items()}
    group_sizes = {name: len(records) for name, records in groups.items()}
    total_images = sum(group_sizes.values())

    def aggregate(selected_names: Sequence[str]) -> tuple[Counter, int]:
        counts = Counter()
        size = 0
        for selected_name in selected_names:
            counts.update(group_counts[selected_name])
            size += group_sizes[selected_name]
        return counts, size

    best = None
    for test_size in range(1, min(4, len(names) - 1) + 1):
        for test_tuple in itertools.combinations(names, test_size):
            remaining = [name for name in names if name not in test_tuple]
            for val_size in range(1, min(3, len(remaining)) + 1):
                for val_tuple in itertools.combinations(remaining, val_size):
                    test_counts, test_images = aggregate(test_tuple)
                    val_counts, val_images = aggregate(val_tuple)
                    test_present = set(test_counts)
                    val_present = set(val_counts)
                    score = (
                        len(test_present & all_classes),
                        len(val_present & all_classes),
                        min(test_images, 1200),
                        min(val_images, 800),
                        -abs(test_images - 0.2 * total_images),
                    )
                    if best is None or score > best[0]:
                        best = (score, set(test_tuple), set(val_tuple))
    if best is None:
        raise ValueError("Could not choose scene groups.")
    return best[1], best[2]


def build_scene_coverage_split(source_root: Path, output_root: Path) -> dict:
    clear_dir(output_root)
    groups = collect_scene_records(source_root)
    test_groups, val_groups = choose_coverage_groups(groups)
    rows = []
    for group, records in groups.items():
        split = "test" if group in test_groups else "valid" if group in val_groups else "train"
        for image, label, classes in records:
            copy_record(image, label, output_root, split)
            rows.append(
                {
                    "group": group,
                    "split": split,
                    "image": str(image),
                    "classes": ";".join(CANONICAL_CLASSES[class_id] for class_id in sorted(classes)),
                }
            )
    write_rows_csv(output_root / "scene_coverage_split_manifest.csv", rows)
    write_data_yaml(output_root)
    summary = {"groups": {}, "splits": {}}
    for group, records in groups.items():
        summary["groups"][group] = {"images": len(records), "box_counts": dict(class_counts(records))}
    for split in ("train", "valid", "test"):
        split_records = [record for group, records in groups.items() if (group in test_groups and split == "test") or (group in val_groups and split == "valid") or (group not in test_groups | val_groups and split == "train") for record in records]
        counts = class_counts(split_records)
        summary["splits"][split] = {
            "groups": sorted(test_groups if split == "test" else val_groups if split == "valid" else set(groups) - test_groups - val_groups),
            "images": len(split_records),
            "box_counts": {CANONICAL_CLASSES[class_id]: counts[class_id] for class_id in CANONICAL_CLASSES},
            "covered_classes": sum(counts[class_id] > 0 for class_id in CANONICAL_CLASSES),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-original", type=Path, required=True)
    parser.add_argument("--external-masked", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(r"LOCAL_PROJECT_ROOT\paper_framework\datasets\protocol_repair_v2"))
    parser.add_argument("--report-root", type=Path, default=Path(r"LOCAL_PROJECT_ROOT\paper_framework\reports\protocol_repair_v2"))
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_dir(args.report_root)
    external = split_external_calibration_locked(
        args.external_masked,
        args.output_root / "external_preprocessed_ffb_calibration_locked",
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    scene = build_scene_coverage_split(
        args.source_original,
        args.output_root / "science_oilpalm_scene_coverage_disjoint",
    )
    payload = {"external_calibration_locked": external, "scene_coverage_disjoint": scene}
    (args.report_root / "protocol_repair_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

