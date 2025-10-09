#!/usr/bin/env python3
"""Visualise raw vs. body-masked prediction masks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_val = arr.min() if arr.size else 0.0
    max_val = arr.max() if arr.size else 1.0
    if max_val <= min_val:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def _load_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("F")
    data = np.array(img, dtype=np.float32)
    if data.max(initial=0.0) > 1.0:
        data /= 255.0
    return data


def _parse_component_replacements(items: Iterable[str]) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid replacement '{item}'. Expected SRC:DST.")
        src, dst = item.split(":", 1)
        replacements[src] = dst
    return replacements


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
    image_path: Path | None,
    ground_truth_path: Path | None,
    dest: Path,
    cmap: str,
    alpha: float,
    outline_alpha: float,
    outline_color: tuple[float, float, float],
    outline_threshold: float,
    outline_thickness: int,
    gt_outline_alpha: float,
    gt_outline_color: tuple[float, float, float],
    gt_outline_threshold: float,
    gt_outline_thickness: int,
) -> None:
    raw = _load_mask(raw_path)
    masked = _load_mask(masked_path)
    gt_mask = _load_mask(ground_truth_path) if ground_truth_path is not None else None

    dest.parent.mkdir(parents=True, exist_ok=True)

    raw_norm = _normalize(raw)
    masked_norm = _normalize(masked)
    diff = masked - raw
    diff_pos = np.clip(diff, 0.0, None)
    diff_norm = _normalize(diff_pos)
    gt_norm = _normalize(gt_mask) if gt_mask is not None else None

    from matplotlib import pyplot as plt

    if image_path is None or not image_path.exists():
        panels = [
            ("Raw Prediction Mask", raw_norm),
            ("Body-masked Prediction", masked_norm),
            ("Masked - Raw", diff_norm),
        ]
        if gt_norm is not None:
            panels.append(("Ground Truth Mask", gt_norm))

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        for axis, (title, data) in zip(np.atleast_1d(axes), panels):
            axis.imshow(data, cmap=cmap)
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(dest, bbox_inches="tight")
        plt.close(fig)
        return

    image = Image.open(image_path).convert("RGB")
    image_arr = np.array(image, dtype=np.float32) / 255.0

    raw_overlay = _overlay_outlines(
        image_arr, raw, outline_color, outline_alpha, outline_threshold, outline_thickness
    )
    masked_overlay = _overlay_outlines(
        image_arr, masked, outline_color, outline_alpha, outline_threshold, outline_thickness
    )
    gt_overlay = None
    if gt_mask is not None:
        gt_overlay = _overlay_outlines(
            image_arr,
            gt_mask,
            gt_outline_color,
            gt_outline_alpha,
            gt_outline_threshold,
            gt_outline_thickness,
        )

    columns = [("Image", image_arr), ("Raw Outline Overlay", raw_overlay), ("Masked Outline Overlay", masked_overlay)]
    bottom = [("Raw Heatmap", raw_norm), ("Masked Heatmap", masked_norm), ("Masked - Raw", diff_norm)]

    if gt_overlay is not None:
        columns.append(("Ground Truth Overlay", gt_overlay))
        bottom.append(("Ground Truth Mask", gt_norm))

    fig, axes = plt.subplots(2, len(columns), figsize=(4 * len(columns), 8))
    axes = np.atleast_2d(axes)

    for idx, (title, data) in enumerate(columns):
        axes[0, idx].imshow(data if data.ndim == 3 else data, cmap=None if data.ndim == 3 else cmap)
        axes[0, idx].set_title(title)
        axes[0, idx].axis("off")

    for idx, (title, data) in enumerate(bottom):
        axes[1, idx].imshow(data if data.ndim == 3 else data, cmap=None if data.ndim == 3 else cmap)
        axes[1, idx].set_title(title)
        axes[1, idx].axis("off")

    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise raw vs. filtered prediction masks.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing raw prediction masks.")
    parser.add_argument("--masked-dir", type=Path, required=True, help="Directory containing body-masked prediction masks.")
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
    parser.add_argument("--gt-outline-alpha", type=float, default=0.85, help="Opacity used when compositing ground-truth outlines.")
    parser.add_argument("--gt-outline-color", type=str, default="0.0,1.0,0.0", help="Ground-truth outline RGB, e.g. '0,255,0'.")
    parser.add_argument("--gt-outline-threshold", type=float, default=0.5, help="Threshold applied when extracting ground-truth outlines.")
    parser.add_argument("--gt-outline-thickness", type=int, default=1, help="Ground-truth outline thickness iterations.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip entries missing raw/masked/image files instead of raising.")
    return parser.parse_args()


def _apply_replacements(relative: Path, replacements: dict[str, str]) -> Path:
    if not replacements:
        return relative
    parts = [replacements.get(part, part) for part in relative.parts]
    return Path(*parts)


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

    alt_exts = {relative.suffix.lower(), ".png"}
    alt_exts.update(IMAGE_EXTENSIONS)

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

    alt_exts = IMAGE_EXTENSIONS | {relative.suffix.lower()}
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

    alt_exts = set(IMAGE_EXTENSIONS)
    alt_exts.add(relative.suffix.lower())

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
        replaced = _apply_replacements(rel, replacements)
        for candidate in _candidate_ground_truth_relatives(replaced):
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
        replaced = _apply_replacements(rel, replacements)
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

    ground_truth_root = args.ground_truth_dir if args.ground_truth_dir is not None else args.image_dir
    if ground_truth_root is not None and "img" not in ground_truth_replacements:
        ground_truth_replacements = {"img": "label", **ground_truth_replacements}

    raw_files = [
        path for path in args.raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not raw_files:
        raise FileNotFoundError(f"No prediction masks found in {args.raw_dir}.")

    outline_color = _parse_outline_color(args.outline_color)
    gt_outline_color = _parse_outline_color(args.gt_outline_color)

    processed = 0
    for raw_path in raw_files:
        relative = raw_path.relative_to(args.raw_dir)
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

        image_path = None
        if args.image_dir is not None:
            image_path = _resolve_image_path(args.image_dir, raw_path, args.raw_dir, image_replacements)

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
                image_path,
                ground_truth_path,
                dest,
                args.cmap,
                args.overlay_alpha,
                args.outline_alpha,
                outline_color,
                args.outline_threshold,
                max(1, args.outline_thickness),
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
