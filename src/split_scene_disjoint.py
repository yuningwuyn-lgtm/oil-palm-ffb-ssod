"""Create a scene/video-disjoint ScienceDB/OILPALM split."""

from __future__ import annotations

import argparse
from pathlib import Path

from full_ssod_ffb_pipeline import (
    convert_to_yolo_format,
    scene_aware_yolo_split,
    set_safe_delete_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene-disjoint YOLO split from a labeled oil-palm dataset.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-groups", default="")
    parser.add_argument("--test-groups", default="")
    args = parser.parse_args()

    set_safe_delete_root(args.output)
    canonical_yaml = convert_to_yolo_format(args.source, args.output / "canonical")
    scene_aware_yolo_split(
        canonical_yaml.parent,
        args.output / "sciencedb_scene_disjoint",
        val_groups_arg=args.val_groups,
        test_groups_arg=args.test_groups,
    )


if __name__ == "__main__":
    main()

