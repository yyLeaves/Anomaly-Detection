#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Iterable
from dataclasses import dataclass
import argparse
import sys
import numpy as np
from PIL import Image

ARRAY_EXTENSIONS = {".npy"}
IMAGE_EXTENSIONS = {".png", ".nifti"}
VALID_EXTENSIONS = ARRAY_EXTENSIONS | IMAGE_EXTENSIONS


@dataclass
class ArrayWithMeta:
    data: np.ndarray
    dtype: np.dtype


def _load_array(path: Path) -> ArrayWithMeta:
    suffix = path.suffix.lower()
    if suffix not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported prediction mask format: {path}")

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
    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"Body mask must be an image file, found: {path}")
    image = Image.open(path).convert("F")
    mask = np.array(image, dtype=np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return (mask >= threshold).astype(np.float32)


def _broadcast_mask(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if len(target_shape) == 2:
        return mask
    if len(target_shape) == 3:
        return np.broadcast_to(mask[..., None], target_shape)
    raise ValueError(f"Unsupported prediction mask shape {target_shape}")


def _apply_mask(pred_mask: ArrayWithMeta, mask: np.ndarray) -> np.ndarray:
    masked = pred_mask.data * mask
    target_dtype = pred_mask.dtype
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        masked = np.clip(masked, info.min, info.max)
        return masked.astype(target_dtype)
    return masked.astype(target_dtype)


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


def apply_body_mask_to_predictions(
    prediction_root: Path,
    mask_root: Path,
    output_root: Path,
    *,
    threshold: float,
    strict: bool,
) -> None:
    pred_files = [
        path
        for path in prediction_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    if not pred_files:
        raise FileNotFoundError(f"No prediction masks found in {prediction_root}.")

    for pred_path in pred_files:
        relative = pred_path.relative_to(prediction_root)
        stem = relative.stem
        if stem.endswith("_pred_mask"):
            stem = stem[: -len("_pred_mask")]

        mask_path = mask_root / relative.parent / "bodymask" / (stem + ".png")

        if not mask_path.exists():
            msg = f"Missing body mask for {pred_path.relative_to(prediction_root)} (expected {mask_path})"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[WARN] {msg}", file=sys.stderr)
            continue

        pred_array = _load_array(pred_path)
        mask = _load_mask(mask_path, threshold=threshold)
        mask_broadcast = _broadcast_mask(mask, pred_array.data.shape)
        masked = _apply_mask(pred_array, mask_broadcast)

        destination = output_root / pred_path.relative_to(prediction_root)
        _save_array(destination, masked)



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply body masks to prediction masks prior to evaluation."
    )
    parser.add_argument("--prediction-mask-dir", type=Path, required=True,
                        help="Directory containing prediction masks.")
    parser.add_argument("--body-mask-dir", type=Path, required=True,
                        help="Directory with body mask images.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination directory for masked prediction masks.")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Threshold for binarizing body mask (default=0.5).")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip missing body masks instead of raising error.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    threshold = args.mask_threshold
    if threshold > 1.0:
        threshold = threshold / 255.0

    apply_body_mask_to_predictions(
        prediction_root=args.prediction_mask_dir,
        mask_root=args.body_mask_dir,
        output_root=args.output_dir,
        threshold=threshold,
        strict=not args.skip_missing,
    )


if __name__ == "__main__":
    main()
