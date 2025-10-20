#!/usr/bin/env python3
"""Visualise raw, body-masked, and filtered prediction masks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from fastflow_postprocess import (
    apply_replacements,
    canonical_suffix,
    load_array,
    load_image_as_rgb,
    normalise_for_display,
    parse_replacements,
    project_to_2d,
)

PREDICTION_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".npy",
    ".npz",
    ".nii",
    ".nii.gz",
}
DATA_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".nii",
    ".nii.gz",
}


def _normalize(arr: np.ndarray) -> np.ndarray:
    return normalise_for_display(arr)


def _load_mask(path: Path) -> np.ndarray:
    array = load_array(path)
    data = array.data.astype(np.float32, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        max_val = np.iinfo(array.dtype).max
        if max_val > 0:
            data = data / float(max_val)
    else:
        max_val = float(np.max(np.abs(data))) if data.size else 1.0
        if max_val > 0:
            data = data / max_val
    return project_to_2d(data)


def _parse_component_replacements(items: Iterable[str]) -> dict[str, str]:
    return parse_replacements(items)


def _binary_outline(mask: np.ndarray, threshold: float, thickness: int) -> np.ndarray:
    """Return edges of mask pixels above threshold."""
    binary = (mask >= threshold).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)

    def erode(arr: np.ndarray) -> np.ndarray:
        padded = np.pad(arr, 1, mode="constant", constant_values=0)
        result = np.ones_like(arr, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[1 + dy : 1 + dy + arr.shape[0], 1 + dx : 1 + dx + arr.shape[1]]
                result &= shifted
        return result

    eroded = binary.copy()
    for _ in range(max(thickness, 1)):
        next_eroded = erode(eroded)
        if not next_eroded.any():
            eroded = next_eroded
            break
        eroded = next_eroded

    outline = binary.astype(bool) & ~eroded.astype(bool)
    return outline


def _overlay_outlines(
    image_arr: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
    threshold: float,
    thickness: int,
) -> np.ndarray:
    overlay = image_arr.copy()
    outline = _binary_outline(mask, threshold=threshold, thickness=thickness)
    if outline.any():
        color_arr = np.array(color, dtype=np.float32)
        overlay[outline] = alpha * color_arr + (1 - alpha) * overlay[outline]
    return overlay.clip(0.0, 1.0)


def save_panel(
    raw_path: Path,
    masked_path: Path,
    body_mask_path: Path | None,
    filtered_path: Path | None,
    image_path: Path | None,
    ground_truth_path: Path | None,
    dest: Path,
    cmap: str,
    alpha: float,
    outline_alpha: float,
    outline_color: tuple[float, float, float],
    outline_threshold: float,
    outline_thickness: int,
    body_outline_alpha: float,
    body_outline_color: tuple[float, float, float],
    body_outline_threshold: float,
    body_outline_thickness: int,
    body_fill_color: tuple[float, float, float],
    body_fill_alpha: float,
    body_fill_threshold: float,
    body_dim_factor: float,
    gt_outline_alpha: float,
    gt_outline_color: tuple[float, float, float],
    gt_outline_threshold: float,
    gt_outline_thickness: int,
) -> None:
    raw = _load_mask(raw_path)
    masked = _load_mask(masked_path)
    body_mask = _load_mask(body_mask_path) if body_mask_path is not None else None
    filtered = _load_mask(filtered_path) if filtered_path is not None else None
    gt_mask = _load_mask(ground_truth_path) if ground_truth_path is not None else None

    dest.parent.mkdir(parents=True, exist_ok=True)

    raw_norm = _normalize(raw)
    masked_norm = _normalize(masked)
    diff = masked - raw
    diff_pos = np.clip(diff, 0.0, None)
    diff_norm = _normalize(diff_pos)

    body_mask_norm = _normalize(body_mask) if body_mask is not None else None

    filtered_norm = None
    filtered_diff_norm = None
    if filtered is not None:
        filtered_norm = _normalize(filtered)
        filtered_diff = filtered - masked
        filtered_diff_pos = np.clip(filtered_diff, 0.0, None)
        filtered_diff_norm = _normalize(filtered_diff_pos)

    gt_norm = _normalize(gt_mask) if gt_mask is not None else None

    from matplotlib import pyplot as plt
    from matplotlib import cm

    final_mask = filtered if filtered is not None else masked
    final_norm = filtered_norm if filtered_norm is not None else masked_norm
    final_label = "Filtered Prediction" if filtered is not None else "Body-masked Prediction"

    base_image: np.ndarray | None = None
    if image_path is not None and image_path.exists():
        base_image = load_image_as_rgb(image_path)

    def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr
        raise ValueError(f"Unexpected array shape for RGB conversion: {arr.shape}")

    def _heat_overlay(base: np.ndarray, mask_norm: np.ndarray) -> np.ndarray:
        heat_rgb = np.asarray(cm.get_cmap(cmap)(mask_norm))[..., :3]
        return np.clip(alpha * heat_rgb + (1.0 - alpha) * base, 0.0, 1.0)

    if base_image is None:
        base_image = _ensure_rgb(np.ones_like(raw_norm))

    raw_base = base_image.copy()

    def _blend_color(base: np.ndarray, mask_norm: np.ndarray, color: tuple[float, float, float], alpha_val: float) -> np.ndarray:
        mask_clipped = np.clip(mask_norm, 0.0, 1.0)
        color_arr = np.array(color, dtype=np.float32)
        weight = np.expand_dims(mask_clipped * alpha_val, axis=-1)
        return np.clip((1.0 - weight) * base + weight * color_arr, 0.0, 1.0)

    raw_overlay = _blend_color(raw_base, raw_norm, (1.0, 0.2, 0.2), 0.9)
    raw_overlay = _overlay_outlines(
        raw_overlay,
        raw,
        (1.0, 0.35, 0.35),
        min(1.0, outline_alpha + 0.05),
        outline_threshold,
        max(1, outline_thickness),
    )

    final_base = base_image.copy()
    if body_mask is not None and body_mask_norm is not None:
        inside = body_mask_norm >= body_fill_threshold
        outside = ~inside
        body_dim_factor = max(0.0, min(1.0, body_dim_factor))
        body_fill_alpha = max(0.0, min(1.0, body_fill_alpha))
        if body_dim_factor < 1.0:
            final_base[outside] = final_base[outside] * body_dim_factor
        if body_fill_alpha > 0.0:
            fill_color = np.array(body_fill_color, dtype=np.float32)
            final_base[inside] = (
                (1.0 - body_fill_alpha) * final_base[inside] + body_fill_alpha * fill_color
            )
        final_base = _overlay_outlines(
            final_base,
            body_mask,
            body_outline_color,
            body_outline_alpha,
            body_outline_threshold,
            max(1, body_outline_thickness),
        )
        if body_outline_thickness >= 2:
            final_base = _overlay_outlines(
                final_base,
                body_mask,
                (0.15, 0.8, 1.0),
                min(1.0, body_outline_alpha + 0.1),
                body_outline_threshold,
                max(1, body_outline_thickness - 1),
            )
    final_overlay = _heat_overlay(final_base, final_norm)
    final_overlay = _overlay_outlines(
        final_overlay,
        final_mask,
        outline_color,
        outline_alpha,
        outline_threshold,
        max(1, outline_thickness),
    )

    if body_mask_norm is not None:
        final_positive = final_mask > 0
        body_interior = body_mask_norm >= body_fill_threshold
        outside_mask = final_positive & (~body_interior)
        if outside_mask.any():
            violation_color = np.array([1.0, 0.85, 0.1], dtype=np.float32)
            final_overlay[outside_mask] = violation_color
            inside_edge = final_positive & body_interior
            if inside_edge.any():
                inside_color = np.array([1.0, 0.25, 0.25], dtype=np.float32)
                final_overlay[inside_edge] = np.clip(
                    0.6 * final_overlay[inside_edge] + 0.4 * inside_color,
                0.0,
                1.0,
            )

    panels: list[tuple[str, np.ndarray]] = [
        ("Raw Prediction Overlay", raw_overlay),
        (f"{final_label} + Body Mask", final_overlay),
    ]

    if gt_norm is not None and gt_mask is not None:
        gt_overlay = _overlay_outlines(
            base_image.copy(),
            gt_mask,
            gt_outline_color,
            gt_outline_alpha,
            gt_outline_threshold,
            max(1, gt_outline_thickness),
        )
        panels.append(("Ground Truth Overlay", gt_overlay))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    axes = np.atleast_1d(axes)
    for axis, (title, data) in zip(axes, panels):
        axis.imshow(data)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise raw vs. filtered prediction masks.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing raw prediction masks.")
    parser.add_argument("--masked-dir", type=Path, required=True, help="Directory containing body-masked prediction masks.")
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        default=None,
        help="Optional directory containing consecutively-filtered prediction masks.",
    )
    parser.add_argument(
        "--body-mask-dir",
        type=Path,
        default=None,
        help="Optional directory containing anatomical body masks for overlays.",
    )
    parser.add_argument(
        "--body-mask-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Path component replacements when resolving body mask paths (e.g. 'img:bodymask').",
    )
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional dataset root for fetching original images.")
    parser.add_argument(
        "--image-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Path component replacements when resolving image paths (e.g. 'prediction_masks:test').",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Optional dataset root containing ground-truth masks (typically the 'label' folders).",
    )
    parser.add_argument(
        "--ground-truth-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Path component replacements for ground-truth lookup (e.g. 'img:label').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where comparison figures will be written.",
    )
    parser.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap used for heatmaps.")
    parser.add_argument("--overlay-alpha", type=float, default=0.6, help="Opacity for heatmap overlays when no image is available.")
    parser.add_argument("--outline-alpha", type=float, default=0.85, help="Opacity applied to outline colour during compositing.")
    parser.add_argument("--outline-color", type=str, default="1.0,0.0,0.0", help="RGB values (0-1 or 0-255) for outline colour, e.g. '255,0,0'.")
    parser.add_argument("--outline-threshold", type=float, default=0.2, help="Threshold applied before extracting mask outlines.")
    parser.add_argument("--outline-thickness", type=int, default=1, help="Outline thickness (iterations of erosion).")
    parser.add_argument("--body-outline-alpha", type=float, default=0.6, help="Opacity used when compositing body-mask outlines.")
    parser.add_argument("--body-outline-color", type=str, default="0.0,0.0,1.0", help="Body-mask outline RGB, e.g. '0,0,255'.")
    parser.add_argument("--body-outline-threshold", type=float, default=0.5, help="Threshold applied when extracting body-mask outlines.")
    parser.add_argument("--body-outline-thickness", type=int, default=1, help="Body-mask outline thickness iterations.")
    parser.add_argument("--body-fill-alpha", type=float, default=0.25, help="Alpha used to tint the interior of the body mask.")
    parser.add_argument("--body-fill-color", type=str, default="0.0,0.2,0.9", help="RGB tint applied inside the body mask.")
    parser.add_argument("--body-fill-threshold", type=float, default=0.5, help="Threshold applied when determining body mask interior.")
    parser.add_argument("--body-dim-factor", type=float, default=0.3, help="Multiplier applied to image intensities outside the body mask.")
    parser.add_argument("--gt-outline-alpha", type=float, default=0.85, help="Opacity used when compositing ground-truth outlines.")
    parser.add_argument("--gt-outline-color", type=str, default="0.0,1.0,0.0", help="Ground-truth outline RGB, e.g. '0,255,0'.")
    parser.add_argument("--gt-outline-threshold", type=float, default=0.5, help="Threshold applied when extracting ground-truth outlines.")
    parser.add_argument("--gt-outline-thickness", type=int, default=1, help="Ground-truth outline thickness iterations.")
    parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        metavar="LABEL",
        help="Dataset labels to include (e.g. 'Ungood'). Defaults to Ungood unless --include-all-labels is set.",
    )
    parser.add_argument(
        "--include-all-labels",
        action="store_true",
        help="Process masks from all dataset labels (disables label filtering).",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip entries missing raw/masked/image files instead of raising.")
    return parser.parse_args()


def _candidate_mask_relatives(relative: Path) -> list[Path]:
    """Generate filenames that could correspond to a masked prediction."""
    parent = relative.parent
    stem_variants = [relative.stem]
    first = stem_variants[0]
    if first.endswith("_pred_mask"):
        base = first[: -len("_pred_mask")]
        stem_variants.extend([base, f"{base}_anomaly_map"])
    elif first.endswith("_anomaly_map"):
        base = first[: -len("_anomaly_map")]
        stem_variants.extend([base, f"{base}_pred_mask"])
    else:
        stem_variants.extend([f"{first}_pred_mask", f"{first}_anomaly_map"])

    alt_exts = {
        canonical_suffix(relative),
        relative.suffix.lower(),
        ".png",
        ".npy",
        ".npz",
        ".nii",
        ".nii.gz",
    }
    alt_exts.update(PREDICTION_EXTENSIONS)
    alt_exts = {ext for ext in alt_exts if ext}

    candidates: list[Path] = []
    seen: set[Path] = set()
    for stem in stem_variants:
        for ext in alt_exts:
            if not ext:
                continue
            candidate = parent / f"{stem}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _resolve_masked_path(root: Path, relative: Path) -> tuple[Path | None, list[Path]]:
    candidates = [root / candidate for candidate in _candidate_mask_relatives(relative)]
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return None, candidates


def _candidate_image_relatives(relative: Path) -> list[Path]:
    """Return possible dataset image filenames derived from a raw mask path."""
    parent = relative.parent
    stem_variants = [relative.stem]
    first = stem_variants[0]
    if first.endswith("_pred_mask"):
        base = first[: -len("_pred_mask")]
        stem_variants.extend([base, f"{base}_anomaly_map"])
    elif first.endswith("_anomaly_map"):
        base = first[: -len("_anomaly_map")]
        stem_variants.extend([base, f"{base}_pred_mask"])
    else:
        stem_variants.append(f"{first}_pred_mask")

    alt_exts = {
        canonical_suffix(relative),
        relative.suffix.lower(),
    }
    alt_exts.update(DATA_IMAGE_EXTENSIONS)
    alt_exts = {ext for ext in alt_exts if ext}
    candidates: list[Path] = []
    seen: set[Path] = set()
    for stem in stem_variants:
        for ext in alt_exts:
            if not ext:
                continue
            candidate = parent / f"{stem}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _candidate_ground_truth_relatives(relative: Path) -> list[Path]:
    parent = relative.parent
    stem_variants = [relative.stem]
    first = stem_variants[0]
    if first.endswith("_pred_mask"):
        base = first[: -len("_pred_mask")]
    elif first.endswith("_anomaly_map"):
        base = first[: -len("_anomaly_map")]
    else:
        base = first
    stem_variants.extend(
        [
            base,
            f"{base}_mask",
            f"{base}_label",
            f"{base}_gt",
        ]
    )

    alt_exts = set(PREDICTION_EXTENSIONS)
    alt_exts.add(canonical_suffix(relative))
    alt_exts.add(relative.suffix.lower())
    alt_exts = {ext for ext in alt_exts if ext}

    candidates: list[Path] = []
    seen: set[Path] = set()
    for stem in stem_variants:
        for ext in alt_exts:
            if not ext:
                continue
            candidate = parent / f"{stem}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _candidate_body_mask_relatives(relative: Path) -> list[Path]:
    parent = relative.parent
    stem_variants = [relative.stem]
    first = stem_variants[0]
    if first.endswith("_pred_mask"):
        base = first[: -len("_pred_mask")]
    elif first.endswith("_anomaly_map"):
        base = first[: -len("_anomaly_map")]
    else:
        base = first
    stem_variants.extend(
        [
            base,
            f"{base}_bodymask",
            f"{base}_mask",
            f"{base}_label",
        ]
    )

    alt_exts = {
        canonical_suffix(relative),
        relative.suffix.lower(),
        ".png",
        ".npy",
        ".npz",
        ".nii",
        ".nii.gz",
    }
    alt_exts = {ext for ext in alt_exts if ext}

    candidates: list[Path] = []
    seen: set[Path] = set()
    for stem in stem_variants:
        for ext in alt_exts:
            candidate = parent / f"{stem}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _resolve_ground_truth_path(
    root: Path,
    raw_path: Path,
    raw_dir: Path,
    replacements: dict[str, str],
) -> Path | None:
    relative_forms = [raw_path.relative_to(raw_dir)]
    try:
        relative_forms.append(raw_path.relative_to(raw_dir.parent))
    except ValueError:
        pass

    candidate_relatives: list[Path] = []
    seen_relatives: set[Path] = set()
    for rel in relative_forms:
        replaced = apply_replacements(rel, replacements)
        for candidate in _candidate_ground_truth_relatives(replaced):
            if candidate not in seen_relatives:
                seen_relatives.add(candidate)
                candidate_relatives.append(candidate)

    for candidate in candidate_relatives:
        path = root / candidate
        if path.exists():
            return path
    return None


def _resolve_body_mask_path(
    root: Path,
    raw_path: Path,
    raw_dir: Path,
    replacements: dict[str, str],
) -> Path | None:
    relative_forms = [raw_path.relative_to(raw_dir)]
    try:
        relative_forms.append(raw_path.relative_to(raw_dir.parent))
    except ValueError:
        pass

    candidate_relatives: list[Path] = []
    seen_relatives: set[Path] = set()
    for rel in relative_forms:
        replaced = apply_replacements(rel, replacements)
        for candidate in _candidate_body_mask_relatives(replaced):
            if candidate not in seen_relatives:
                seen_relatives.add(candidate)
                candidate_relatives.append(candidate)

    for candidate in candidate_relatives:
        path = root / candidate
        if path.exists():
            return path
    return None


def _resolve_image_path(
    root: Path,
    raw_path: Path,
    raw_dir: Path,
    replacements: dict[str, str],
) -> Path | None:
    relative_forms = [
        raw_path.relative_to(raw_dir),
    ]
    try:
        relative_forms.append(raw_path.relative_to(raw_dir.parent))
    except ValueError:
        pass

    candidate_relatives: list[Path] = []
    seen_relatives: set[Path] = set()
    for rel in relative_forms:
        replaced = apply_replacements(rel, replacements)
        for candidate in _candidate_image_relatives(replaced):
            if candidate not in seen_relatives:
                seen_relatives.add(candidate)
                candidate_relatives.append(candidate)

    for relative in candidate_relatives:
        path = root / relative
        if path.exists():
            return path
    return None


def _parse_outline_color(color_str: str) -> tuple[float, float, float]:
    parts = color_str.split(",")
    if len(parts) != 3:
        raise ValueError(f"Outline color '{color_str}' must contain three comma-separated values.")
    values: list[float] = []
    for part in parts:
        try:
            value = float(part.strip())
        except ValueError:
            raise ValueError(f"Invalid outline color component '{part}' in '{color_str}'.") from None
        if value > 1.0:
            value = value / 255.0
        values.append(min(max(value, 0.0), 1.0))
    return values[0], values[1], values[2]


def main() -> None:
    args = parse_args()

    image_replacements = _parse_component_replacements(args.image_replace)
    ground_truth_replacements = _parse_component_replacements(args.ground_truth_replace)
    body_mask_replacements = _parse_component_replacements(args.body_mask_replace)

    ground_truth_root = args.ground_truth_dir if args.ground_truth_dir is not None else args.image_dir
    if ground_truth_root is not None and "img" not in ground_truth_replacements:
        ground_truth_replacements = {"img": "label", **ground_truth_replacements}

    if args.body_mask_dir is not None:
        defaults = {"img": "bodymask", "prediction_masks": "bodymask"}
        defaults.update(body_mask_replacements)
        body_mask_replacements = defaults

    raw_files = [
        path
        for path in args.raw_dir.rglob("*")
        if path.is_file() and canonical_suffix(path) in PREDICTION_EXTENSIONS
    ]
    if not raw_files:
        raise FileNotFoundError(f"No prediction masks found in {args.raw_dir}.")

    outline_color = _parse_outline_color(args.outline_color)
    gt_outline_color = _parse_outline_color(args.gt_outline_color)
    body_outline_color = _parse_outline_color(args.body_outline_color)
    body_fill_color = _parse_outline_color(args.body_fill_color)

    if args.include_all_labels:
        include_labels: set[str] | None = None
    else:
        include_labels = {label for label in (args.include_label or ["Ungood"])}

    processed = 0
    for raw_path in raw_files:
        relative = raw_path.relative_to(args.raw_dir)
        if include_labels is not None:
            parts = {part for part in relative.parts}
            if include_labels.isdisjoint(parts):
                continue

        masked_path, candidates = _resolve_masked_path(args.masked_dir, relative)
        if masked_path is None:
            preview = ", ".join(str(path) for path in candidates[:5])
            message = (
                f"Missing masked prediction for {relative} "
                f"(searched {len(candidates)} candidates; examples: {preview})"
            )
            if args.skip_missing:
                print(f"[WARN] {message}")
                continue
            raise FileNotFoundError(message)

        filtered_path = None
        if args.filtered_dir is not None:
            filtered_path, filtered_candidates = _resolve_masked_path(args.filtered_dir, relative)
            if filtered_path is None:
                preview = ", ".join(str(path) for path in filtered_candidates[:5])
                message = (
                    f"Missing filtered prediction for {relative} "
                    f"(searched {len(filtered_candidates)} candidates; examples: {preview})"
                )
                if args.skip_missing:
                    print(f"[WARN] {message}")
                else:
                    print(f"[WARN] {message}; continuing without filtered stage.")

        image_path = None
        if args.image_dir is not None:
            image_path = _resolve_image_path(args.image_dir, raw_path, args.raw_dir, image_replacements)

        body_mask_path = None
        if args.body_mask_dir is not None:
            body_mask_path = _resolve_body_mask_path(args.body_mask_dir, raw_path, args.raw_dir, body_mask_replacements)
            if body_mask_path is None:
                message = (
                    f"Missing body mask for {relative} using root {args.body_mask_dir}. "
                    "Continuing without body-mask overlays."
                )
                if args.skip_missing:
                    print(f"[WARN] {message}")
                else:
                    print(f"[WARN] {message}")

        ground_truth_path = None
        if ground_truth_root is not None:
            ground_truth_path = _resolve_ground_truth_path(
                ground_truth_root, raw_path, args.raw_dir, ground_truth_replacements
            )
            if ground_truth_path is None and not args.skip_missing:
                print(
                    f"[WARN] Missing ground-truth mask for {relative}; continuing without it.",
                    flush=True,
                )

        dest = args.output_dir / relative.with_suffix(".png")
        try:
            save_panel(
                raw_path,
                masked_path,
                body_mask_path,
                filtered_path,
                image_path,
                ground_truth_path,
                dest,
                args.cmap,
                args.overlay_alpha,
                args.outline_alpha,
                outline_color,
                args.outline_threshold,
                max(1, args.outline_thickness),
                args.body_outline_alpha,
                body_outline_color,
                args.body_outline_threshold,
                max(1, args.body_outline_thickness),
                body_fill_color,
                args.body_fill_alpha,
                args.body_fill_threshold,
                args.body_dim_factor,
                args.gt_outline_alpha,
                gt_outline_color,
                args.gt_outline_threshold,
                max(1, args.gt_outline_thickness),
            )
            processed += 1
        except Exception as exc:
            message = f"Failed to render {relative}: {exc}"
            if args.skip_missing:
                print(f"[WARN] {message}")
                continue
            raise

    print(f"[INFO] Generated {processed} prediction comparison figures in {args.output_dir}")


if __name__ == "__main__":
    main()
