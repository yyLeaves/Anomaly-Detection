#!/usr/bin/env python3
"""Utilities for preparing FastFlow datasets (PNG or NIfTI)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from PIL import Image

from fastflow_postprocess import canonical_suffix, normalise_for_display

NIFTI_SUFFIXES = {".nii", ".nii.gz"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NUMPY_SUFFIXES = {".npy", ".npz"}
POTENTIAL_MASK_COMPONENTS = {"bodymask", "mask", "label", "labels"}


def detect_dataset_format(root: Path, *, scan_limit: int = 200) -> str:
    """Return 'nifti' if any NIfTI files are found, otherwise 'png'."""
    count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = canonical_suffix(path)
        if suffix in NIFTI_SUFFIXES:
            return "nifti"
        if suffix in IMAGE_SUFFIXES or suffix in NUMPY_SUFFIXES:
            count += 1
        if count >= scan_limit:
            break
    return "png"


def _is_mask_path(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    if POTENTIAL_MASK_COMPONENTS & lower_parts:
        return True
    stem = path.stem.lower()
    return any(token in stem for token in ("mask", "label", "seg"))


def _normalise_image_array(data: np.ndarray) -> np.ndarray:
    """Normalise floating-point image arrays to uint8 in [0, 255]."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if data.ndim == 1:
        return normalise_for_display(data[np.newaxis, :])[0]  # pragma: no cover
    if data.ndim == 2:
        norm = normalise_for_display(data)
        return (norm * 255.0).clip(0, 255).astype(np.uint8)
    if data.ndim >= 3:
        data = np.squeeze(data)
        if data.ndim == 2:
            return _normalise_image_array(data)
        if data.ndim > 3:
            first_two = data.shape[:2]
            trailing = int(np.prod(data.shape[2:]))
            data = data.reshape(*first_two, trailing)
        channels = []
        depth = min(data.shape[-1], 3)
        for idx in range(depth):
            norm = normalise_for_display(data[..., idx])
            channels.append((norm * 255.0).clip(0, 255).astype(np.uint8))
        if depth == 1:
            return channels[0]
        return np.stack(channels, axis=-1)
    raise ValueError(f"Unsupported image data shape {data.shape}")


def _normalise_mask_array(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim >= 3:
        data = data[..., 0]
    binary = (data >= 0.5).astype(np.uint8)
    return (binary * 255).astype(np.uint8)


def _convert_nifti_file(source: Path, destination: Path) -> None:
    is_mask = _is_mask_path(source)
    image = nib.load(str(source))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if is_mask:
        payload = _normalise_mask_array(data)
    else:
        payload = _normalise_image_array(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(payload).save(destination)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _iter_source_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def convert_nifti_tree_to_png(source_root: Path, dest_root: Path) -> None:
    """Mirror a NIfTI dataset tree as PNG images for FastFlow."""
    for source in _iter_source_files(source_root):
        suffix = canonical_suffix(source)
        relative = source.relative_to(source_root)
        if suffix in NIFTI_SUFFIXES:
            dest = dest_root / relative.with_suffix(".png")
            if dest.exists() and dest.stat().st_mtime >= source.stat().st_mtime:
                continue
            _convert_nifti_file(source, dest)
        elif suffix in IMAGE_SUFFIXES or suffix in NUMPY_SUFFIXES:
            dest = dest_root / relative
            if dest.exists() and dest.stat().st_mtime >= source.stat().st_mtime:
                continue
            _copy_file(source, dest)
        else:
            # Skip non-image artefacts but preserve directory structure implicitly.
            continue


def prepare_dataset_root(
    data_root: Path,
    *,
    format_hint: str = "auto",
    cache_root: Path | None = None,
) -> tuple[Path, bool]:
    """Return a dataset root that FastFlow can load (PNG tree).

    When `data_root` contains NIfTI volumes, they are converted to PNGs under a cache directory.
    Returns a tuple of `(prepared_root, converted)` where `converted` indicates whether conversion happened.
    """
    if format_hint not in {"auto", "png", "nifti"}:
        raise ValueError(f"Unknown dataset format hint '{format_hint}'")

    resolved_root = data_root.resolve()
    if format_hint == "png":
        return resolved_root, False

    if format_hint == "nifti" or detect_dataset_format(resolved_root) == "nifti":
        cache_dir = cache_root or resolved_root.parent / f"{resolved_root.name}_png_cache"
        cache_dir = cache_dir.resolve()
        convert_nifti_tree_to_png(resolved_root, cache_dir)
        return cache_dir, True

    return resolved_root, False
