#!/usr/bin/env python3
"""End-to-end post-processing pipeline for OOD model prediction masks.

This script strings together the existing building blocks:
1. Apply body masks to the extracted PNG prediction masks.
2. Run the morphology processor (early small-component filtering + dilate/erode rounds).
3. Remove regions that do not persist across consecutive slices.
4. Reconstruct both the raw and the post-processed masks into 3D NIfTI volumes.

Each stage mirrors the folder layout of the input prediction masks so downstream
analysis (metrics, visualisations) can re-use the same relative paths.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence
from dataclasses import asdict
import json

from apply_bodymask import apply_body_mask
from postprocess_utils import parse_replacements
from filter_prediction_masks_consecutive import filter_prediction_masks
from morphology.processor import BatchProcessor, MorphologyProcessor
from morphology.stack_to_3d import BatchNIfTIStacker
from evaluate_model_outputs import evaluate_outputs


logger = logging.getLogger("post_processing_pipeline")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _gather_leaf_dirs(root: Path, extensions: Iterable[str]) -> list[Path]:
    """Return sorted list of directories that contain at least one file with the desired suffixes."""
    candidates: set[Path] = set()
    for suffix in extensions:
        for file in root.rglob(f"*{suffix}"):
            if file.is_file():
                candidates.add(file.parent)
    return sorted(candidates, key=lambda item: str(item))


def _infer_anomaly_map_dir(prediction_dir: Path) -> Path | None:
    """Attempt to infer the anomaly-map directory from the prediction directory structure."""
    parts = list(prediction_dir.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "prediction_masks":
            candidate_parts = parts.copy()
            candidate_parts[idx] = "anomaly_maps"
            candidate = Path(*candidate_parts)
            if candidate.exists():
                return candidate
            break
    return None


def _run_body_mask_stage(
    input_dir: Path,
    body_mask_dir: Path,
    output_dir: Path,
    *,
    mask_threshold: float,
    replacements: dict[str, str],
    extra_mask_roots: list[Path],
    skip_missing: bool,
) -> None:
    logger.info("Applying body masks to prediction slices → %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replacements = dict(replacements)
    if "img" not in replacements:
        replacements = {"img": "bodymask", **replacements}

    split_hint: str | None = None
    for ancestor in (input_dir,) + tuple(input_dir.parents):
        name_lower = ancestor.name.lower()
        if name_lower in {"train", "valid", "test"}:
            split_hint = ancestor.name
            break

    primary_mask_root = body_mask_dir
    auto_extra_roots: list[Path] = []

    if split_hint:
        split_path = body_mask_dir / split_hint
        if split_path.exists():
            primary_mask_root = split_path
            auto_extra_roots.append(body_mask_dir)
    else:
        for candidate in ("train", "valid", "test"):
            split_path = body_mask_dir / candidate
            if split_path.exists():
                auto_extra_roots.append(split_path)

    auto_extra_roots.extend(extra_mask_roots)

    unique_extra: list[Path] = []
    seen: set[Path] = set()
    for path in auto_extra_roots:
        resolved = path.resolve()
        if resolved == primary_mask_root.resolve():
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique_extra.append(path)

    apply_body_mask(
        anomaly_root=input_dir,
        mask_root=primary_mask_root,
        output_root=output_dir,
        threshold=mask_threshold,
        replacements=replacements,
        strict=not skip_missing,
        prediction_root=None,
        raw_prediction_root=None,
        extra_mask_roots=unique_extra,
    )


def _run_morphology_stage(
    input_dir: Path,
    output_dir: Path,
    *,
    processor: MorphologyProcessor,
    binarize_threshold: float,
    max_samples: int | None,
    verbose: bool,
) -> None:
    logger.info("Running morphology processor on body-masked slices → %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = BatchProcessor(processor)
    mask_dirs = _gather_leaf_dirs(input_dir, extensions=(".png", ".npy"))
    if not mask_dirs:
        raise FileNotFoundError(f"No PNG/NPY masks found under {input_dir}")

    for idx, source_dir in enumerate(mask_dirs, start=1):
        relative = source_dir.relative_to(input_dir)
        destination = output_dir / relative
        destination.mkdir(parents=True, exist_ok=True)
        logger.debug("Morphology [%d/%d]: %s → %s", idx, len(mask_dirs), source_dir, destination)
        batch.process_directory(
            input_dir=source_dir,
            output_dir=destination,
            binarize_threshold=binarize_threshold,
            max_samples=max_samples,
            verbose=verbose,
        )


def _run_consecutive_filter_stage(input_dir: Path, output_dir: Path) -> int:
    logger.info("Filtering anomalies that do not persist across slices → %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kept = filter_prediction_masks(input_dir, output_dir)
    logger.info("Saved consecutive-slice filtered masks for %d slices", kept)
    return kept


def _stack_png_tree_to_nifti(
    input_dir: Path,
    output_dir: Path,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    patient_pattern: str,
    verbose: bool,
) -> None:
    logger.info("Reconstructing PNG slices into NIfTI volumes → %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stacker = BatchNIfTIStacker(spacing=spacing, origin=origin, patient_pattern=patient_pattern)
    mask_dirs = stacker.find_mask_directories(input_dir)
    if not mask_dirs:
        logger.warning("No PNG slices found under %s – skipping NIfTI reconstruction.", input_dir)
        return

    for idx, mask_dir in enumerate(mask_dirs, start=1):
        relative = mask_dir.relative_to(input_dir)
        destination = output_dir / relative
        destination.mkdir(parents=True, exist_ok=True)
        stats = stacker.process_directory(mask_dir, destination, verbose=verbose)
        logger.debug(
            "Stacked %d/%d directories (%s) → %s | patients=%d, slices=%d, success=%d, failed=%d",
            idx,
            len(mask_dirs),
            mask_dir,
            destination,
            stats.get("patients", 0),
            stats.get("slices", 0),
            stats.get("success", 0),
            stats.get("failed", 0),
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full PNG-based post-processing pipeline for prediction masks.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with extracted prediction PNG masks.")
    parser.add_argument("--body-mask-dir", type=Path, required=True, help="Dataset directory containing body masks.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory where intermediate and final outputs will be written.",
    )
    parser.add_argument(
        "--path-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Optional path component replacements used to resolve body-mask locations (e.g. 'img:bodymask').",
    )
    parser.add_argument(
        "--extra-mask-dir",
        action="append",
        default=[],
        type=Path,
        help="Additional directories to search for body masks. May be supplied multiple times.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Optional root containing ground-truth masks. When provided, corresponding 3D NIfTI volumes are created.",
    )
    parser.add_argument(
        "--anomaly-map-dir",
        type=Path,
        default=None,
        help="Optional directory with raw anomaly maps for AUROC benchmarking (defaults to auto-detection).",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold applied before binarising body masks (values >1 are treated as 0-255 inputs).",
    )
    parser.add_argument(
        "--skip-missing-body-mask",
        action="store_true",
        help="Skip slices without body masks instead of raising an error during the body-mask stage.",
    )
    # Morphology configuration.
    parser.add_argument("--binarize-threshold", type=float, default=0.5, help="Threshold used inside MorphologyProcessor.")
    parser.add_argument("--dilate-iterations", type=int, default=1, help="Number of dilation iterations per round.")
    parser.add_argument("--erode-iterations", type=int, default=1, help="Number of erosion iterations per round.")
    parser.add_argument("--num-rounds", type=int, default=1, help="Number of dilation+erosion rounds.")
    parser.add_argument("--kernel-size", type=int, default=5, help="Size of the morphology structuring element (must be odd).")
    parser.add_argument(
        "--kernel-shape",
        choices={"ellipse", "rect"},
        default="ellipse",
        help="Shape of the morphology structuring element.",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=3,
        help="Minimum component size (in pixels) that survives the early filtering stage.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices={4, 8},
        default=8,
        help="Connectivity used when labelling connected components.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of masks processed per directory (useful for dry runs).",
    )
    # Volume reconstruction parameters.
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Voxel spacing (in mm) used when constructing NIfTI volumes.",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("OX", "OY", "OZ"),
        help="Origin encoded in the NIfTI affine.",
    )
    parser.add_argument(
        "--patient-pattern",
        type=str,
        default=r"PA(\d+)_",
        help="Regex used to extract patient IDs from filenames when stacking slices.",
    )
    parser.add_argument(
        "--metrics-ground-truth-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Replacements used when locating ground-truth masks for metrics (default adds 'img:label').",
    )
    parser.add_argument(
        "--metrics-mean-fraction-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.1],
        help="Mean positive fraction thresholds evaluated at the patient level.",
    )
    parser.add_argument(
        "--metrics-prediction-threshold",
        type=float,
        default=None,
        help="Optional threshold applied before binarising predictions when computing metrics.",
    )
    parser.add_argument(
        "--metrics-ground-truth-threshold",
        type=float,
        default=None,
        help="Optional threshold applied before binarising ground-truth masks when computing metrics.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (specify once for INFO, twice for DEBUG).",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    body_mask_dir = args.body_mask_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    anomaly_map_dir: Path | None = None
    if args.anomaly_map_dir is not None:
        anomaly_map_dir = args.anomaly_map_dir.resolve()
    elif args.ground_truth_dir is not None:
        inferred = _infer_anomaly_map_dir(input_dir)
        if inferred is not None:
            logger.info("Auto-detected anomaly-map directory for AUROC benchmarks: %s", inferred)
            anomaly_map_dir = inferred

    mask_threshold = args.mask_threshold
    replacements = parse_replacements(args.path_replace)
    extra_mask_roots = [path.resolve() for path in args.extra_mask_dir]

    # Prepare output directories.
    body_masked_dir = output_root / "01_body_masked_png"
    morph_dir = output_root / "02_morphology_png"
    filtered_dir = output_root / "03_consecutive_filtered_png"
    volumes_root = output_root / "volumes"
    raw_volumes_dir = volumes_root / "raw"
    processed_volumes_dir = volumes_root / "post_processed"
    ground_truth_volumes_dir = volumes_root / "ground_truth"

    # 0) Preserve the original PNG predictions as 3D NIfTI volumes.
    _stack_png_tree_to_nifti(
        input_dir=input_dir,
        output_dir=raw_volumes_dir,
        spacing=tuple(args.spacing),
        origin=tuple(args.origin),
        patient_pattern=args.patient_pattern,
        verbose=args.verbose >= 2,
    )

    # 1) Apply body masks.
    _run_body_mask_stage(
        input_dir=input_dir,
        body_mask_dir=body_mask_dir,
        output_dir=body_masked_dir,
        mask_threshold=mask_threshold,
        replacements=replacements,
        extra_mask_roots=extra_mask_roots,
        skip_missing=args.skip_missing_body_mask,
    )

    # 2) Morphology (includes early small-component filtering).
    processor = MorphologyProcessor(
        dilate_iterations=args.dilate_iterations,
        erode_iterations=args.erode_iterations,
        num_rounds=args.num_rounds,
        kernel_size=args.kernel_size,
        kernel_shape=args.kernel_shape,
        min_component_size=args.min_component_size,
        connectivity=args.connectivity,
    )
    _run_morphology_stage(
        input_dir=body_masked_dir,
        output_dir=morph_dir,
        processor=processor,
        binarize_threshold=args.binarize_threshold,
        max_samples=args.max_samples,
        verbose=args.verbose >= 2,
    )

    # 3) Keep only anomalies that persist across consecutive slices.
    _run_consecutive_filter_stage(morph_dir, filtered_dir)

    # 4) Reconstruct post-processed masks into 3D volumes.
    _stack_png_tree_to_nifti(
        input_dir=filtered_dir,
        output_dir=processed_volumes_dir,
        spacing=tuple(args.spacing),
        origin=tuple(args.origin),
        patient_pattern=args.patient_pattern,
        verbose=args.verbose >= 2,
    )

    if args.ground_truth_dir is not None:
        ground_truth_dir = args.ground_truth_dir.resolve()
        _stack_png_tree_to_nifti(
            input_dir=ground_truth_dir,
            output_dir=ground_truth_volumes_dir,
            spacing=tuple(args.spacing),
            origin=tuple(args.origin),
            patient_pattern=args.patient_pattern,
            verbose=args.verbose >= 2,
        )

        metrics_dir = output_root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "metrics_summary.json"

        metrics_replacements = parse_replacements(args.metrics_ground_truth_replace)
        if "img" not in metrics_replacements:
            metrics_replacements = {"img": "label", **metrics_replacements}

        print("\n[INFO] Metrics on POST-PROCESSED masks:")
        evaluation = evaluate_outputs(
            morph_dir,
            ground_truth_dir,
            ground_truth_replacements=metrics_replacements,
            prediction_threshold=args.metrics_prediction_threshold,
            ground_truth_threshold=args.metrics_ground_truth_threshold,
            mean_fraction_thresholds=args.metrics_mean_fraction_thresholds,
            print_summary=True,
            anomaly_map_dir=anomaly_map_dir,
        )

        print("\n[INFO] Metrics on RAW masks (pre post-processing):")
        raw_evaluation = evaluate_outputs(
            input_dir,
            ground_truth_dir,
            ground_truth_replacements=metrics_replacements,
            prediction_threshold=args.metrics_prediction_threshold,
            ground_truth_threshold=args.metrics_ground_truth_threshold,
            mean_fraction_thresholds=args.metrics_mean_fraction_thresholds,
            print_summary=True,
            anomaly_map_dir=anomaly_map_dir,
        )

        payload = {
            "post_processed": {
                "source": "post_processed",
                "prediction_dir": str(morph_dir),
                "pixel_metrics": evaluation["pixel_metrics"],
                "slice_image_metrics": evaluation["slice_image_metrics"],
                "patient_mean_fraction_metrics": evaluation["patient_mean_fraction_metrics"],
                "patient_summary": evaluation["patient_summary"],
                "per_slice": [asdict(record) for record in evaluation["per_slice"]],
                "anomaly_auroc_metrics": evaluation["anomaly_auroc_metrics"],
            },
            "raw_prediction": {
                "source": "raw",
                "prediction_dir": str(input_dir),
                "pixel_metrics": raw_evaluation["pixel_metrics"],
                "slice_image_metrics": raw_evaluation["slice_image_metrics"],
                "patient_mean_fraction_metrics": raw_evaluation["patient_mean_fraction_metrics"],
                "patient_summary": raw_evaluation["patient_summary"],
                "per_slice": [asdict(record) for record in raw_evaluation["per_slice"]],
                "anomaly_auroc_metrics": raw_evaluation["anomaly_auroc_metrics"],
            },
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved metrics summary to %s", metrics_path)

    logger.info("Pipeline completed successfully.")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    run_pipeline(args)


if __name__ == "__main__":
    main()
