#!/usr/bin/env python3
"""Filter PNG prediction masks to keep only anomalies persisting across slices."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import label


@dataclass(frozen=True)
class SliceInfo:
    slice_index: int
    source_path: Path
    relative_path: Path


def _parse_slice_index(stem: str) -> tuple[str, int] | None:
    """Return (volume_id, slice_id) extracted from '<volume>_<slice>' style names."""
    if stem.endswith("_pred_mask"):
        stem = stem[: -len("_pred_mask")]
    if "_" not in stem:
        return None
    volume_id, slice_token = stem.rsplit("_", 1)
    try:
        slice_index = int(slice_token)
    except ValueError:
        return None
    return volume_id, slice_index


def _load_png_mask(path: Path) -> np.ndarray:
    """Load a mask PNG and return a boolean array."""
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def _save_png_mask(path: Path, mask: np.ndarray) -> None:
    """Save a boolean mask as 0/224 PNG."""
    payload = (mask.astype(np.uint8)) * 224
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(payload).save(path)


def _group_png_slices(input_root: Path) -> dict[tuple[Path, str], list[SliceInfo]]:
    """Collect PNG slices grouped by relative directory and volume id."""
    grouped: dict[tuple[Path, str], list[SliceInfo]] = defaultdict(list)
    for png_path in input_root.rglob("*.png"):
        if not png_path.is_file():
            continue
        relative_path = png_path.relative_to(input_root)
        volume_key = _parse_slice_index(png_path.stem)
        if volume_key is None:
            print(f"[WARN] Skipping file with unrecognised naming pattern: {png_path}", file=sys.stderr)
            continue
        volume_id, slice_index = volume_key
        group_key = (relative_path.parent, volume_id)
        grouped[group_key].append(SliceInfo(slice_index, png_path, relative_path))
    return grouped


def _filter_volume_slices(slices: Iterable[SliceInfo], output_root: Path) -> int:
    """Filter connected regions that do not persist across neighbouring slices."""
    ordered = sorted(slices, key=lambda info: info.slice_index)
    if not ordered:
        return 0

    masks = [_load_png_mask(info.source_path) for info in ordered]
    filtered_count = 0
    structure = np.ones((3, 3), dtype=int)

    for idx, (slice_info, mask) in enumerate(zip(ordered, masks)):
        labeled_mask, num_components = label(mask.astype(np.uint8), structure=structure)
        keep = np.zeros_like(mask, dtype=bool)

        prev_mask = masks[idx - 1] if idx > 0 else None
        next_mask = masks[idx + 1] if idx + 1 < len(masks) else None

        for component_id in range(1, num_components + 1):
            region = labeled_mask == component_id
            if not region.any():
                continue

            keep_region = False
            if prev_mask is not None and np.any(prev_mask & region):
                keep_region = True
            if not keep_region and next_mask is not None and np.any(next_mask & region):
                keep_region = True

            if keep_region:
                keep |= region

        destination = output_root / slice_info.relative_path
        _save_png_mask(destination, keep)

        if keep.any():
            filtered_count += 1

    return filtered_count


def filter_prediction_masks(input_root: Path, output_root: Path) -> int:
    """Apply consecutive-slice filtering to all PNG masks under input_root."""
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    grouped = _group_png_slices(input_root)
    total_kept = 0
    for (_relative_parent, volume_id), slices in grouped.items():
        kept = _filter_volume_slices(slices, output_root)
        total_kept += kept
    return total_kept


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only PNG prediction mask regions that persist across consecutive slices."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing post-processed prediction mask PNGs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for filtered masks.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    total = filter_prediction_masks(args.input_dir.resolve(), args.output_dir.resolve())
    print(f"[INFO] Saved filtered PNG masks for {total} slices to {args.output_dir}")


if __name__ == "__main__":
    main()
