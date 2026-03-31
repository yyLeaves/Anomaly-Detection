#!/usr/bin/env python3
"""CLI to visualise anomaly maps next to thresholded outputs."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from postprocess_utils import (
    parse_replacements,
    visualize_anomaly_thresholded_pairs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise anomaly maps side-by-side with thresholded outputs."
    )
    parser.add_argument(
        "--anomaly-dir",
        type=Path,
        required=True,
        help="Directory containing anomaly maps.",
    )
    parser.add_argument(
        "--thresholded-dir",
        type=Path,
        default=None,
        help="Optional directory with thresholded outputs/prediction masks.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used for binarisation (or when thresholding anomaly maps directly).",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        required=True,
        help="Destination for side-by-side comparison PNGs.",
    )
    parser.add_argument(
        "--comparison-cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap used for anomaly map visualisation.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional directory containing source images for overlay panels.",
    )
    parser.add_argument(
        "--image-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Path component replacements when resolving image paths (e.g. 'anomaly_maps:test').",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.6,
        help="Alpha blending factor used for optional anomaly overlays.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip samples with no matching thresholded output instead of raising an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image_replacements = parse_replacements(args.image_replace)

    count = visualize_anomaly_thresholded_pairs(
        anomaly_root=args.anomaly_dir,
        thresholded_root=args.thresholded_dir,
        comparison_root=args.comparison_dir,
        threshold=args.threshold,
        comparison_cmap=args.comparison_cmap,
        image_root=args.image_dir,
        image_replacements=image_replacements,
        overlay_alpha=args.overlay_alpha,
        skip_missing=args.skip_missing,
    )

    if args.thresholded_dir is None:
        print(
            f"[INFO] Generated {count} anomaly-vs-thresholded comparisons in "
            f"{args.comparison_dir} (thresholded on the fly at {args.threshold})."
        )
    else:
        print(
            f"[INFO] Generated {count} anomaly-vs-thresholded comparisons in "
            f"{args.comparison_dir} using thresholded inputs from {args.thresholded_dir}."
        )


if __name__ == "__main__":
    main()
