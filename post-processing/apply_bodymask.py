#!/usr/bin/env python3
"""Post-processing utilities for OOD model anomaly maps.

This module applies body masks to per-image anomaly maps prior to evaluation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from postprocess_utils import (
    apply_mask,
    apply_replacements,
    broadcast_mask,
    canonical_pred_mask_name,
    canonical_suffix,
    is_supported_file,
    load_array,
    load_body_mask,
    parse_replacements,
    save_array,
)


def _find_mask_by_name(mask_root: Path, filename: str) -> Path | None:
    """Fallback search for a mask file by filename anywhere under mask_root."""
    try:
        return next(mask_root.rglob(filename))
    except StopIteration:
        return None


def _resolve_mask_path(
    anomaly_path: Path,
    anomaly_root: Path,
    mask_root: Path,
    replacements: dict[str, str],
) -> Path:
    relative = anomaly_path.relative_to(anomaly_root)
    mapped_relative = apply_replacements(relative, replacements)
    return mask_root / mapped_relative


def apply_body_mask(
    anomaly_root: Path,
    mask_root: Path,
    output_root: Path,
    *,
    threshold: float,
    replacements: dict[str, str],
    strict: bool,
    prediction_root: Path | None = None,
    raw_prediction_root: Path | None = None,
    extra_mask_roots: list[Path] | None = None,
) -> None:
    mask_search_roots = [mask_root]
    if extra_mask_roots:
        mask_search_roots.extend(extra_mask_roots)
    anomaly_files = [
        path for path in anomaly_root.rglob("*") if path.is_file() and is_supported_file(path)
    ]
    if not anomaly_files:
        raise FileNotFoundError(f"No anomaly maps found in {anomaly_root}.")

    for anomaly_path in anomaly_files:
        relative = apply_replacements(anomaly_path.relative_to(anomaly_root), replacements)

        candidates: list[Path] = []
        for base in mask_search_roots:
            base_mask_path = base / relative
            candidates.append(base_mask_path)
            # When dealing with full patient scans, allow fallback to the original Ungood masks.
            lowered_parts = [part.lower() for part in base_mask_path.parts]
            if "ungood_whole_patient_scans" in lowered_parts:
                replacement_parts = []
                replaced = False
                for part in base_mask_path.parts:
                    if not replaced and part.lower() == "ungood_whole_patient_scans":
                        replacement_parts.append("Ungood")
                        replaced = True
                    else:
                        replacement_parts.append(part)
                fallback_path = Path(*replacement_parts)
                if fallback_path not in candidates:
                    candidates.append(fallback_path)

            stem_variants = {base_mask_path.stem}
            base_stem = base_mask_path.stem
            if base_stem.endswith("_anomaly_map"):
                stem_variants.add(base_stem[: -len("_anomaly_map")])
            if base_stem.endswith("_pred_mask"):
                stem_variants.add(base_stem[: -len("_pred_mask")])

            alt_extensions = {
                ext
                for ext in {
                    canonical_suffix(base_mask_path),
                    base_mask_path.suffix.lower(),
                    ".png",
                    ".npy",
                    ".npz",
                    ".nii",
                    ".nii.gz",
                }
                if ext
            }

            for stem_variant in stem_variants:
                for ext in alt_extensions:
                    candidates.append(base_mask_path.with_name(f"{stem_variant}{ext}"))

        resolved_mask_path = next((cand for cand in candidates if cand.exists()), None)
        fallback_checked: list[str] = []
        if resolved_mask_path is None:
            # As a fallback, search by filename across known subdirectories (good/Ungood).
            for search_root in mask_search_roots:
                fallback = _find_mask_by_name(search_root, (search_root / relative).name)
                if fallback is not None:
                    resolved_mask_path = fallback
                    fallback_checked.append(str(fallback))
                    break

        if resolved_mask_path is None:
            message = (
                f"Missing body mask for {anomaly_path.relative_to(anomaly_root)}. "
                f"Checked: {[str(c) for c in candidates] + fallback_checked}"
            )
            if strict:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}", file=sys.stderr)
            continue
        mask_path = resolved_mask_path

        anomaly_array = load_array(anomaly_path)
        mask = load_body_mask(mask_path, threshold=threshold)
        mask_broadcast = broadcast_mask(mask, anomaly_array.data.shape)
        masked = apply_mask(anomaly_array, mask_broadcast)

        destination = output_root / anomaly_path.relative_to(anomaly_root)
        save_array(destination, masked, template=anomaly_array)

        if prediction_root is not None and raw_prediction_root is not None:
            relative = anomaly_path.relative_to(anomaly_root)

            stem_variants = {relative.stem}
            if relative.stem.endswith("_anomaly_map"):
                base = relative.stem[: -len("_anomaly_map")]
                stem_variants.update({base, f"{base}_pred_mask"})
            elif not relative.stem.endswith("_pred_mask"):
                stem_variants.add(f"{relative.stem}_pred_mask")

            suffix_candidates = {
                canonical_suffix(relative),
                relative.suffix.lower(),
                ".png",
                ".npy",
                ".npz",
                ".nii",
                ".nii.gz",
            }
            suffix_candidates.discard("")

            raw_candidates: list[Path] = []
            for stem_variant in stem_variants:
                for ext in suffix_candidates:
                    candidate = raw_prediction_root / relative.with_name(f"{stem_variant}{ext}")
                    raw_candidates.append(candidate)

            seen: set[Path] = set()
            unique_candidates: list[Path] = []
            for candidate in raw_candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    unique_candidates.append(candidate)

            raw_prediction_path = next((cand for cand in unique_candidates if cand.exists()), None)
            if raw_prediction_path is None:
                print(
                    f"[WARN] Missing raw prediction mask for {relative} "
                    f"(checked {[str(c) for c in unique_candidates]}).",
                    file=sys.stderr,
                )
            else:
                raw_prediction_array = load_array(raw_prediction_path)
                prediction_mask = broadcast_mask(mask, raw_prediction_array.data.shape)
                masked_prediction = apply_mask(raw_prediction_array, prediction_mask)
                try:
                    prediction_relative = raw_prediction_path.relative_to(raw_prediction_root)
                except ValueError:
                    prediction_relative = relative
                suffix = canonical_suffix(prediction_relative) or canonical_suffix(raw_prediction_path)
                prediction_relative = canonical_pred_mask_name(prediction_relative, suffix=suffix)
                prediction_path = prediction_root / prediction_relative
                save_array(prediction_path, masked_prediction, template=raw_prediction_array)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply body masks to anomaly maps prior to evaluation."
    )
    parser.add_argument("--anomaly-dir", type=Path, required=True, help="Directory containing anomaly maps.")
    parser.add_argument("--body-mask-dir", type=Path, required=True, help="Directory with body mask images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for masked anomaly maps.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold (0-255 for 8-bit masks) applied before binarising the mask.",
    )
    parser.add_argument(
        "--path-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Optional path component replacement (e.g. 'img:bodymask'). Can be supplied multiple times.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip samples without body masks instead of raising an error.",
    )
    parser.add_argument(
        "--prediction-mask-dir",
        type=Path,
        default=None,
        help="Optional output directory for body-masked prediction masks.",
    )
    parser.add_argument(
        "--raw-prediction-dir",
        type=Path,
        default=None,
        help="Directory containing the original prediction masks (defaults to swapping 'anomaly_maps' with 'prediction_masks' in --anomaly-dir).",
    )
    parser.add_argument(
        "--extra-mask-root",
        action="append",
        default=[],
        type=Path,
        help="Additional directories to search for body masks (e.g. dataset/test/good/bodymask). Can be supplied multiple times.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    replacements = parse_replacements(args.path_replace)
    threshold = args.mask_threshold
    if threshold > 1.0:
        threshold = threshold / 255.0

    prediction_root = args.prediction_mask_dir
    raw_prediction_root = args.raw_prediction_dir
    if prediction_root is not None and raw_prediction_root is None:
        anomaly_dir = Path(args.anomaly_dir)
        parts = list(anomaly_dir.parts)
        if "anomaly_maps" in parts:
            idx = parts.index("anomaly_maps")
            parts[idx] = "prediction_masks"
            raw_prediction_root = Path(*parts)
        else:
            raw_prediction_root = anomaly_dir.parent / "prediction_masks"

    apply_body_mask(
        anomaly_root=args.anomaly_dir,
        mask_root=args.body_mask_dir,
        output_root=args.output_dir,
        threshold=threshold,
        replacements=replacements,
        strict=not args.skip_missing,
        prediction_root=prediction_root,
        raw_prediction_root=raw_prediction_root,
        extra_mask_roots=args.extra_mask_root or [],
    )

    message = f"[INFO] Applied body masks to anomaly maps in {args.anomaly_dir} -> {args.output_dir}"
    if prediction_root is not None:
        message += f"; prediction masks saved to {prediction_root}"
    print(message)


if __name__ == "__main__":
    main()
