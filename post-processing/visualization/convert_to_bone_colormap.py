#!/usr/bin/env python3
"""Convert a NIfTI dataset to PNGs coloured with the 'bone' colormap."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib import cm
try:  # Matplotlib ≥ 3.6
    from matplotlib import colormaps as mpl_colormaps
except ImportError:  # Older Matplotlib
    mpl_colormaps = None
from PIL import Image

from postprocess_utils import canonical_suffix, normalise_for_display

POTENTIAL_MASK_COMPONENTS = {"bodymask", "mask", "label", "labels"}
NIFTI_SUFFIXES = {".nii", ".nii.gz"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NUMPY_SUFFIXES = {".npy", ".npz"}
if mpl_colormaps is not None:
    BONE_CMAP = mpl_colormaps.get_cmap("bone")
else:  # pragma: no cover
    BONE_CMAP = cm.get_cmap("bone")


def _looks_like_mask(path: Path) -> bool:
    parts_lower = {part.lower() for part in path.parts}
    if POTENTIAL_MASK_COMPONENTS & parts_lower:
        return True
    stem = path.stem.lower()
    return any(token in stem for token in ("mask", "label", "seg"))


def _normalise_mask_array(data: np.ndarray) -> np.ndarray:
    payload = np.asarray(data, dtype=np.float32)
    payload = np.squeeze(payload)
    if payload.ndim >= 3:
        payload = payload[..., 0]
    binary = (payload >= 0.5).astype(np.uint8)
    return (binary * 255).astype(np.uint8)


def _coerce_to_2d(data: np.ndarray) -> np.ndarray:
    payload = np.asarray(data, dtype=np.float32)
    payload = np.squeeze(payload)
    if payload.ndim == 0:
        return np.zeros((1, 1), dtype=np.float32)
    if payload.ndim == 1:
        return payload[np.newaxis, :]
    if payload.ndim > 2:
        first_two = payload.shape[:2]
        trailing = int(np.prod(payload.shape[2:]))
        payload = payload.reshape(first_two[0], first_two[1], trailing)
        payload = payload[..., 0]
    return payload


def _apply_bone_colormap(data: np.ndarray) -> np.ndarray:
    payload = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    scalar = _coerce_to_2d(payload)
    norm = normalise_for_display(scalar)
    rgba = BONE_CMAP(norm)
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return rgb


def _convert_nifti_file(source: Path, destination: Path) -> None:
    image = nib.load(str(source))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if _looks_like_mask(source):
        payload = _normalise_mask_array(data)
    else:
        payload = _apply_bone_colormap(data)

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(payload).save(destination)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def convert_dataset(source_root: Path, dest_root: Path) -> None:
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        suffix = canonical_suffix(source)
        relative = source.relative_to(source_root)
        if suffix in NIFTI_SUFFIXES:
            destination = dest_root / relative.with_suffix(".png")
            if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
                continue
            _convert_nifti_file(source, destination)
        elif suffix in IMAGE_SUFFIXES or suffix in NUMPY_SUFFIXES:
            destination = dest_root / relative
            if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
                continue
            _copy_file(source, destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a NIfTI dataset tree into PNGs coloured with the 'bone' colormap."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Root directory with the original NIfTI dataset.")
    parser.add_argument(
        "--dest-root",
        type=Path,
        required=True,
        help="Destination directory where the bone-colormap PNG tree will be created.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    dest_root.mkdir(parents=True, exist_ok=True)
    convert_dataset(source_root, dest_root)
    print(f"[INFO] Finished bone colormap conversion → {dest_root}")


if __name__ == "__main__":
    main()
