#!/usr/bin/env python3
"""CLI to visualise original vs. body-masked anomaly maps."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from apply_bodymask import (
    _parse_replacements,
    visualize_anomaly_pairs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise original vs. body-masked anomaly maps."
    )
    parser.add_argument(
        "--anomaly-dir",
        type=Path,
        required=True,
        help="Directory containing the original (unmasked) anomaly maps.",
    )
    parser.add_argument(
        "--masked-dir",
        type=Path,
        required=True,
        help="Directory containing the already masked anomaly maps.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Root directory containing the source images.",
    )
    parser.add_argument(
        "--image-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Path component replacements when resolving image paths (e.g. 'anomaly_maps:test').",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        required=True,
        help="Destination for side-by-side comparison panels.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Optional directory where anomaly overlays (raw/masked) will be stored.",
    )
    parser.add_argument(
        "--comparison-cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap used for heatmaps in the comparison panels.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.6,
        help="Alpha blending factor when creating overlay images.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip samples without a matching masked map instead of raising an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image_replacements = _parse_replacements(args.image_replace)

    count = visualize_anomaly_pairs(
        anomaly_root=args.anomaly_dir,
        masked_root=args.masked_dir,
        image_root=args.image_dir,
        image_replacements=image_replacements,
        comparison_root=args.comparison_dir,
        comparison_cmap=args.comparison_cmap,
        overlay_root=args.overlay_dir,
        overlay_alpha=args.overlay_alpha,
        skip_missing=args.skip_missing,
    )

    message = (
        f"[INFO] Generated visualisations for {count} anomaly map pairs. "
        f"Comparison panels saved to {args.comparison_dir}"
    )
    if args.overlay_dir:
        message += f"; overlays stored in {args.overlay_dir}"
    print(message)


if __name__ == "__main__":
    main()
