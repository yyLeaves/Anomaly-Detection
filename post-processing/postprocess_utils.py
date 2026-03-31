#!/usr/bin/env python3
"""Shared post-processing utilities for anomaly maps and prediction masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NUMPY_EXTENSIONS = {".npy", ".npz"}
NIFTI_EXTENSIONS = {".nii", ".nii.gz"}
VALID_EXTENSIONS = IMAGE_EXTENSIONS | NUMPY_EXTENSIONS | NIFTI_EXTENSIONS


@dataclass
class ArrayWithMeta:
    """Container that keeps float data, the original dtype, and optional metadata."""

    data: np.ndarray
    dtype: np.dtype
    meta: dict[str, object] | None = None


def canonical_suffix(path: Path) -> str:
    """Return a case-insensitive suffix, collapsing compound suffixes like '.nii.gz'."""
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] == [".nii", ".gz"]:
        return ".nii.gz"
    return suffixes[-1] if suffixes else ""


def is_supported_file(path: Path) -> bool:
    """Check whether the file has a recognised extension."""
    return canonical_suffix(path) in VALID_EXTENSIONS


def parse_replacements(replacements: Iterable[str]) -> dict[str, str]:
    """Parse CLI `SRC:DST` pairs into a mapping."""
    mapping: dict[str, str] = {}
    for item in replacements:
        if ":" not in item:
            raise ValueError(f"Invalid replacement '{item}'. Expected format 'src:dst'.")
        src, dst = item.split(":", 1)
        mapping[src] = dst
    return mapping

# Backwards compatibility for modules that imported the private helper.
_parse_replacements = parse_replacements


def apply_replacements(relative_path: Path, mapping: Mapping[str, str]) -> Path:
    """Replace individual path components using `mapping`."""
    if not mapping:
        return relative_path
    parts = [mapping.get(part, part) for part in relative_path.parts]
    return Path(*parts)


def load_array(path: Path) -> ArrayWithMeta:
    """Load anomaly maps, prediction masks, or body masks as float32 arrays."""
    suffix = canonical_suffix(path)
    if suffix not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {path}")

    if suffix in NUMPY_EXTENSIONS:
        array = np.load(path)
        if isinstance(array, np.lib.npyio.NpzFile):
            dataset = array.get("arr_0")
            if dataset is None:
                raise ValueError(f"NPZ file {path} does not contain 'arr_0'.")
            array = dataset
        array = np.asarray(array)
        return ArrayWithMeta(data=array.astype(np.float32, copy=False), dtype=array.dtype, meta={"source": "numpy"})

    if suffix in IMAGE_EXTENSIONS:
        with Image.open(path) as image:
            array = np.asarray(image)
            meta = {"mode": image.mode}
        return ArrayWithMeta(data=array.astype(np.float32, copy=False), dtype=array.dtype, meta=meta)

    # NIfTI volumes
    image = nib.load(str(path))
    dtype = np.dtype(image.get_data_dtype())
    data = np.asarray(image.get_fdata(dtype=np.float32))
    meta = {"affine": image.affine, "header": image.header.copy()}
    return ArrayWithMeta(data=data, dtype=dtype, meta=meta)


def load_body_mask(path: Path, *, threshold: float) -> np.ndarray:
    """Load a body mask and binarise it using the provided threshold."""
    mask_array = load_array(path)
    mask = mask_array.data
    if mask.ndim > 2:
        # Many body masks are stored as RGB(A); use the first colour channel instead of maxing with alpha.
        if mask.ndim == 3 and mask.shape[-1] <= 4:
            mask = mask[..., 0]
        else:
            squeezed = np.squeeze(mask)
            if squeezed.ndim == 3 and squeezed.shape[-1] <= 4:
                mask = squeezed[..., 0]
            else:
                mask = squeezed
    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[0], mask.shape[1], -1)[..., 0]
    if np.issubdtype(mask_array.dtype, np.integer):
        max_val = np.iinfo(mask_array.dtype).max
        if max_val > 0:
            mask = mask / float(max_val)
    if mask.ndim > 0:
        binary = (mask >= threshold).astype(np.float32, copy=False)
    else:
        binary = np.array([mask >= threshold], dtype=np.float32)
    return binary


def broadcast_mask(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Broadcast a binary mask to match the anomaly/prediction tensor shape."""
    mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.shape == tuple(target_shape):
        return mask_array

    if mask_array.ndim < len(target_shape):
        expand_dims = len(target_shape) - mask_array.ndim
        mask_array = mask_array.reshape(mask_array.shape + (1,) * expand_dims)

    try:
        broadcast = np.broadcast_to(mask_array, target_shape)
    except ValueError as exc:
        raise ValueError(f"Cannot broadcast mask of shape {mask.shape} to target shape {tuple(target_shape)}") from exc
    return np.asarray(broadcast, dtype=np.float32)


def apply_mask(anomaly: ArrayWithMeta, mask: np.ndarray) -> np.ndarray:
    """Multiply an anomaly/prediction tensor by a broadcast body mask."""
    masked = anomaly.data * mask
    target_dtype = anomaly.dtype
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        masked = np.clip(masked, info.min, info.max)
        return masked.astype(target_dtype)
    return masked.astype(target_dtype)


def save_array(path: Path, data: np.ndarray, *, template: ArrayWithMeta | None = None) -> None:
    """Save arrays/images/volumes, preserving metadata from `template` when available."""
    suffix = canonical_suffix(path)
    if suffix not in VALID_EXTENSIONS:
        raise ValueError(f"Cannot save unsupported file type: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    target_dtype = template.dtype if template is not None else data.dtype

    if suffix == ".npy":
        np.save(path, data.astype(target_dtype))
        return
    if suffix == ".npz":
        np.savez_compressed(path, data.astype(target_dtype))
        return
    if suffix in IMAGE_EXTENSIONS:
        if not np.issubdtype(target_dtype, np.integer):
            # Default to 8-bit visualisation if the template was not integer-based.
            target_dtype = np.uint8
            normalised = normalise_for_display(data)
            payload = (normalised * 255.0).clip(0, 255).astype(target_dtype)
        else:
            info = np.iinfo(target_dtype)
            payload = np.clip(data, info.min, info.max).astype(target_dtype)
        Image.fromarray(payload).save(path)
        return

    # NIfTI output
    affine = np.eye(4)
    header = None
    if template is not None and template.meta is not None:
        affine = np.asarray(template.meta.get("affine", affine))
        header = template.meta.get("header")
    nib.save(nib.Nifti1Image(data.astype(target_dtype), affine=affine, header=header), str(path))


def project_to_2d(array: np.ndarray) -> np.ndarray:
    """Collapse higher-dimensional arrays to 2D by removing singleton dims and averaging channels."""
    projection = np.asarray(array)
    while projection.ndim > 2:
        if projection.shape[-1] == 1:
            projection = projection[..., 0]
        else:
            projection = projection.mean(axis=-1)
    return projection


def normalise_for_display(array: np.ndarray) -> np.ndarray:
    """Normalise array values to the [0, 1] interval for visualisation."""
    data = np.asarray(array, dtype=np.float32)
    if data.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(data.min()) if data.size else 0.0
    max_val = float(data.max()) if data.size else 0.0
    if max_val <= min_val:
        return np.zeros_like(data, dtype=np.float32)
    return (data - min_val) / (max_val - min_val)


def load_image_as_rgb(path: Path) -> np.ndarray:
    """Load image/volume as an RGB array in the [0, 1] interval."""
    suffix = canonical_suffix(path)
    if suffix in IMAGE_EXTENSIONS:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            data = np.asarray(rgb, dtype=np.float32) / 255.0
        return data

    array = load_array(path)
    data = array.data
    if np.issubdtype(array.dtype, np.integer):
        max_val = np.iinfo(array.dtype).max
        if max_val > 0:
            data = data / float(max_val)
    else:
        max_val = float(np.max(np.abs(data))) if data.size else 1.0
        if max_val > 0:
            data = data / max_val

    projected = project_to_2d(data)
    projected = normalise_for_display(projected)
    return np.stack([projected] * 3, axis=-1)


def make_overlay(
    image_path: Path,
    heatmap: np.ndarray,
    *,
    cmap: str,
    alpha: float,
) -> np.ndarray:
    """Overlay a heatmap on top of an image."""
    from matplotlib import cm

    base = load_image_as_rgb(image_path)
    heat = normalise_for_display(heatmap)
    cmap_fn = cm.get_cmap(cmap)
    heat_rgb = np.asarray(cmap_fn(heat))[..., :3]
    overlay = alpha * heat_rgb + (1.0 - alpha) * base
    return overlay.clip(0.0, 1.0)


def _comparison_figure(
    panels: Sequence[tuple[str, np.ndarray]],
    *,
    cmap: str,
) -> "Image.Image":
    """Generate a comparison panel figure and return it as a PIL image."""
    from matplotlib import pyplot as plt

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    axes = np.atleast_1d(axes)
    for axis, (title, data) in zip(axes, panels):
        if data.ndim == 3 and data.shape[-1] == 3:
            axis.imshow(data)
        else:
            axis.imshow(data, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = np.ascontiguousarray(rgba[..., :3])
    else:
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return Image.fromarray(buf)


def visualize_anomaly_pairs(
    *,
    anomaly_root: Path,
    masked_root: Path,
    image_root: Path,
    image_replacements: Mapping[str, str],
    comparison_root: Path,
    comparison_cmap: str,
    overlay_root: Path | None,
    overlay_alpha: float,
    skip_missing: bool,
) -> int:
    """Create side-by-side comparisons (and optional overlays) for anomaly map pairs."""
    comparison_root.mkdir(parents=True, exist_ok=True)
    if overlay_root is not None:
        overlay_root.mkdir(parents=True, exist_ok=True)

    anomaly_files = sorted(
        path for path in anomaly_root.rglob("*") if path.is_file() and is_supported_file(path)
    )
    if not anomaly_files:
        raise FileNotFoundError(f"No anomaly maps found in {anomaly_root}.")

    processed = 0
    for anomaly_path in anomaly_files:
        relative = anomaly_path.relative_to(anomaly_root)
        masked_path = masked_root / relative
        if not masked_path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Masked anomaly map missing for {relative}. Expected at {masked_path}")

        raw = load_array(anomaly_path)
        masked = load_array(masked_path)

        raw_2d = project_to_2d(raw.data)
        masked_2d = project_to_2d(masked.data)
        diff = masked_2d - raw_2d

        raw_norm = normalise_for_display(raw_2d)
        masked_norm = normalise_for_display(masked_2d)
        diff_norm = normalise_for_display(diff)

        image_relative = apply_replacements(relative, image_replacements)
        image_path = image_root / image_relative
        overlay_panel: np.ndarray | None = None
        if image_path.exists():
            overlay_panel = make_overlay(image_path, masked_2d, cmap=comparison_cmap, alpha=overlay_alpha)

        panels: list[tuple[str, np.ndarray]] = [
            ("Original Anomaly", raw_norm),
            ("Body-masked Anomaly", masked_norm),
            ("Masked - Original", diff_norm),
        ]
        if overlay_panel is not None:
            panels.append(("Overlay", overlay_panel))

        figure = _comparison_figure(panels, cmap=comparison_cmap)
        comparison_path = comparison_root / relative.with_suffix(".png")
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        figure.save(comparison_path)

        if overlay_root is not None and overlay_panel is not None:
            overlay_path = overlay_root / relative.with_suffix(".png")
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((overlay_panel * 255.0).astype(np.uint8)).save(overlay_path)

        processed += 1

    return processed


def _candidate_thresholded_relatives(relative: Path) -> list[Path]:
    """Generate likely filenames for thresholded outputs matching an anomaly-map path."""
    parent = relative.parent
    stem_variants = [relative.stem]
    first = stem_variants[0]
    if first.endswith("_anomaly_map"):
        base = first[: -len("_anomaly_map")]
        stem_variants.extend([base, f"{base}_pred_mask"])
    elif first.endswith("_pred_mask"):
        base = first[: -len("_pred_mask")]
        stem_variants.extend([base, f"{base}_anomaly_map"])
    else:
        stem_variants.extend([f"{first}_pred_mask", f"{first}_anomaly_map"])

    ext_priority = [
        canonical_suffix(relative),
        relative.suffix.lower(),
        ".png",
        ".npy",
        ".npz",
        ".nii",
        ".nii.gz",
    ]
    for ext in sorted(VALID_EXTENSIONS):
        if ext not in ext_priority:
            ext_priority.append(ext)
    alt_exts = [ext for ext in ext_priority if ext]

    candidates: list[Path] = []
    seen: set[Path] = set()
    for stem in stem_variants:
        for ext in alt_exts:
            candidate = parent / f"{stem}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _resolve_thresholded_path(root: Path, relative: Path) -> tuple[Path | None, list[Path]]:
    candidates = [root / candidate for candidate in _candidate_thresholded_relatives(relative)]
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return None, candidates


def _binarise_for_display(data: np.ndarray, *, source_dtype: np.dtype, threshold: float) -> np.ndarray:
    """Convert a thresholded output to a binary map in [0, 1] for display."""
    arr = np.asarray(data, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    unique = np.unique(arr)
    if unique.size <= 3 and float(np.min(arr)) >= 0.0:
        return (arr > 0.0).astype(np.float32, copy=False)

    if np.issubdtype(source_dtype, np.integer):
        info = np.iinfo(source_dtype)
        scaled = threshold * float(info.max) if threshold <= 1.0 else threshold
        return (arr >= scaled).astype(np.float32, copy=False)

    return (arr >= threshold).astype(np.float32, copy=False)


def visualize_anomaly_thresholded_pairs(
    *,
    anomaly_root: Path,
    comparison_root: Path,
    threshold: float,
    comparison_cmap: str,
    thresholded_root: Path | None = None,
    image_root: Path | None = None,
    image_replacements: Mapping[str, str] | None = None,
    overlay_alpha: float = 0.6,
    skip_missing: bool = False,
) -> int:
    """Create side-by-side figures for anomaly maps and their thresholded outputs."""
    comparison_root.mkdir(parents=True, exist_ok=True)
    replacements = image_replacements or {}

    anomaly_files = sorted(
        path for path in anomaly_root.rglob("*") if path.is_file() and is_supported_file(path)
    )
    if not anomaly_files:
        raise FileNotFoundError(f"No anomaly maps found in {anomaly_root}.")

    processed = 0
    for anomaly_path in anomaly_files:
        relative = anomaly_path.relative_to(anomaly_root)
        anomaly = load_array(anomaly_path)
        anomaly_2d = project_to_2d(anomaly.data)
        anomaly_norm = normalise_for_display(anomaly_2d)

        thresholded_path: Path | None = None
        if thresholded_root is not None:
            thresholded_path, candidates = _resolve_thresholded_path(thresholded_root, relative)
            if thresholded_path is None:
                if skip_missing:
                    continue
                preview = ", ".join(str(path) for path in candidates[:5])
                suffix = " ..." if len(candidates) > 5 else ""
                raise FileNotFoundError(
                    f"Missing thresholded output for {relative}. Tried: {preview}{suffix}"
                )
            thresholded = load_array(thresholded_path)
            thresholded_2d = project_to_2d(thresholded.data)
            thresholded_bin = _binarise_for_display(
                thresholded_2d, source_dtype=thresholded.dtype, threshold=threshold
            )
        else:
            thresholded_bin = (anomaly_2d >= threshold).astype(np.float32, copy=False)

        thresholded_rgb = np.stack([thresholded_bin] * 3, axis=-1)
        panels: list[tuple[str, np.ndarray]] = [
            ("Anomaly Map", anomaly_norm),
            ("Thresholded Output", thresholded_rgb),
        ]

        if image_root is not None:
            image_relative = apply_replacements(relative, replacements)
            image_path = image_root / image_relative
            if image_path.exists():
                overlay = make_overlay(
                    image_path,
                    anomaly_2d,
                    cmap=comparison_cmap,
                    alpha=overlay_alpha,
                )
                panels.append(("Anomaly Overlay", overlay))

        figure = _comparison_figure(panels, cmap=comparison_cmap)
        comparison_path = comparison_root / relative.with_suffix(".png")
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        figure.save(comparison_path)
        processed += 1

    return processed


def canonical_pred_mask_name(path: Path, *, suffix: str | None = None) -> Path:
    """Ensure prediction mask filenames end with '_pred_mask' and keep a desired suffix."""
    stem = path.stem
    if stem.endswith("_pred_mask"):
        canonical_stem = stem
    elif stem.endswith("_anomaly_map"):
        canonical_stem = f"{stem[: -len('_anomaly_map')]}_pred_mask"
    else:
        canonical_stem = f"{stem}_pred_mask"
    final_suffix = suffix or canonical_suffix(path) or path.suffix
    if final_suffix == "":
        final_suffix = ".png"
    return path.with_name(f"{canonical_stem}{final_suffix}")
