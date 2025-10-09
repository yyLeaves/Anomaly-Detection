#!/usr/bin/env python3
"""Post-processing utilities for FastFlow anomaly maps.

This module applies body masks to per-image anomaly maps prior to evaluation.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

ARRAY_EXTENSIONS = {".npy", ".npz"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VALID_EXTENSIONS = ARRAY_EXTENSIONS | IMAGE_EXTENSIONS


@dataclass
class ArrayWithMeta:
    """Container that tracks array data and its original dtype."""

    data: np.ndarray
    dtype: np.dtype


def _load_array(path: Path) -> ArrayWithMeta:
    """Load an anomaly map from disk."""
    suffix = path.suffix.lower()
    if suffix not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported anomaly map format: {path}")

    if suffix in ARRAY_EXTENSIONS:
        array = np.load(path)
        if isinstance(array, np.lib.npyio.NpzFile):
            arr = array.get("arr_0")
            if arr is None:
                raise ValueError(f"NPZ file {path} does not contain 'arr_0'.")
            array = arr
    else:
        image = Image.open(path)
        array = np.array(image)

    return ArrayWithMeta(data=array.astype(np.float32, copy=False), dtype=array.dtype)


def _load_mask(path: Path, threshold: float) -> np.ndarray:
    """Load body mask and convert to a binary tensor."""
    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"Body mask must be an image file, found: {path}")
    image = Image.open(path).convert("F")
    mask = np.array(image, dtype=np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    binary_mask = (mask >= threshold).astype(np.float32)
    return binary_mask


def _broadcast_mask(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast mask to match anomaly map shape."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if len(target_shape) == 2:
        return mask
    if len(target_shape) == 3:
        return np.broadcast_to(mask[..., None], target_shape)
    raise ValueError(f"Unsupported anomaly map shape {target_shape} for mask application")


def _apply_mask(anomaly_map: ArrayWithMeta, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask to anomaly map."""
    masked = anomaly_map.data * mask
    target_dtype = anomaly_map.dtype
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        masked = np.clip(masked, info.min, info.max)
        return masked.astype(target_dtype)
    return masked.astype(target_dtype)


def _parse_replacements(replacements: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in replacements:
        if ":" not in item:
            raise ValueError(f"Invalid replacement '{item}'. Expected format 'src:dst'.")
        src, dst = item.split(":", 1)
        mapping[src] = dst
    return mapping


def _apply_replacements(relative_path: Path, mapping: dict[str, str]) -> Path:
    if not mapping:
        return relative_path
    parts = [mapping.get(part, part) for part in relative_path.parts]
    return Path(*parts)


def _resolve_mask_path(
    anomaly_path: Path,
    anomaly_root: Path,
    mask_root: Path,
    replacements: dict[str, str],
) -> Path:
    relative = anomaly_path.relative_to(anomaly_root)
    mapped_relative = _apply_replacements(relative, replacements)
    return mask_root / mapped_relative


def _save_array(path: Path, masked: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, masked)
    elif suffix == ".npz":
        np.savez_compressed(path, masked)
    else:
        image = Image.fromarray(masked)
        image.save(path)


def _canonical_pred_mask_name(path: Path) -> Path:
    """Return a filename whose stem ends with '_pred_mask' and '.png' suffix."""
    stem = path.stem
    if stem.endswith("_pred_mask"):
        canonical_stem = stem
    elif stem.endswith("_anomaly_map"):
        canonical_stem = f"{stem[: -len('_anomaly_map')]}_pred_mask"
    else:
        canonical_stem = f"{stem}_pred_mask"
    return path.with_name(f"{canonical_stem}.png")


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
) -> None:
    anomaly_files = [
        path
        for path in anomaly_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    if not anomaly_files:
        raise FileNotFoundError(f"No anomaly maps found in {anomaly_root}.")

    for anomaly_path in anomaly_files:
        mask_path = _resolve_mask_path(anomaly_path, anomaly_root, mask_root, replacements)
        candidates = [mask_path]

        stem_variants = [mask_path.stem]
        for suffix in ("_anomaly_map", "_pred_mask"):
            if stem_variants[0].endswith(suffix):
                stem_variants.append(stem_variants[0][: -len(suffix)])

        alt_extensions = {mask_path.suffix.lower(), ".png"}

        expanded_candidates = []
        for stem_variant in stem_variants:
            for ext in alt_extensions:
                if not ext:
                    continue
                expanded_candidates.append(mask_path.with_name(f"{stem_variant}{ext}"))

        candidates.extend(expanded_candidates)

        resolved_mask_path = next((cand for cand in candidates if cand.exists()), None)
        if resolved_mask_path is None:
            message = (
                f"Missing body mask for {anomaly_path.relative_to(anomaly_root)}. "
                f"Checked: {[str(c) for c in candidates]}"
            )
            if strict:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}", file=sys.stderr)
            continue
        mask_path = resolved_mask_path

        anomaly_array = _load_array(anomaly_path)
        mask = _load_mask(mask_path, threshold=threshold)
        mask_broadcast = _broadcast_mask(mask, anomaly_array.data.shape)
        masked = _apply_mask(anomaly_array, mask_broadcast)

        destination = output_root / anomaly_path.relative_to(anomaly_root)
        _save_array(destination, masked)

        if prediction_root is not None and raw_prediction_root is not None:
            relative = destination.relative_to(output_root)
            raw_relative = relative.with_suffix(".png")
            stem_variants = [raw_relative.stem]
            if stem_variants[0].endswith("_anomaly_map"):
                stem_variants.extend(
                    [
                        stem_variants[0][: -len("_anomaly_map")],
                        stem_variants[0][: -len("_anomaly_map")] + "_pred_mask",
                    ]
                )
            elif stem_variants[0].endswith("_pred_mask"):
                stem_variants.append(stem_variants[0][: -len("_pred_mask")])

            alt_exts = {raw_relative.suffix.lower(), ".png"}
            raw_candidates: list[Path] = []
            for stem_variant in stem_variants:
                for ext in alt_exts:
                    candidate = raw_relative.with_name(f"{stem_variant}{ext}")
                    raw_candidates.append(raw_prediction_root / candidate)

            # Preserve insertion order while dropping duplicates.
            seen: set[Path] = set()
            unique_raw_candidates: list[Path] = []
            for cand in raw_candidates:
                if cand not in seen:
                    seen.add(cand)
                    unique_raw_candidates.append(cand)

            raw_prediction_path = next((cand for cand in unique_raw_candidates if cand.exists()), None)
            if raw_prediction_path is None:
                print(
                    f"[WARN] Missing raw prediction mask for {relative} (checked {[str(c) for c in unique_raw_candidates]}).",
                    file=sys.stderr,
                )
            else:
                raw_pred = Image.open(raw_prediction_path).convert("F")
                raw_arr = np.array(raw_pred, dtype=np.float32) / 255.0
                if raw_arr.ndim == 3:
                    raw_arr = raw_arr.mean(axis=0)
                if raw_arr.shape != mask.shape:
                    raw_arr = np.resize(raw_arr, mask.shape)
                masked_prediction = raw_arr * mask
                try:
                    prediction_relative = raw_prediction_path.relative_to(raw_prediction_root)
                except ValueError:
                    prediction_relative = raw_relative
                prediction_relative = _canonical_pred_mask_name(prediction_relative)
                prediction_path = prediction_root / prediction_relative
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((masked_prediction * 255.0).astype(np.uint8)).save(prediction_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply body masks to FastFlow anomaly maps prior to evaluation."
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    replacements = _parse_replacements(args.path_replace)
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
    )

    message = f"[INFO] Applied body masks to anomaly maps in {args.anomaly_dir} -> {args.output_dir}"
    if prediction_root is not None:
        message += f"; prediction masks saved to {prediction_root}"
    print(message)


if __name__ == "__main__":
    main()
