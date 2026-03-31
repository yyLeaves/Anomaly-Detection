#!/usr/bin/env python3
"""Compute pixel-level metrics between prediction masks and ground truth."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from postprocess_utils import (
    apply_replacements,
    canonical_suffix,
    load_array,
    project_to_2d,
)
from morphology.slice_metrics import aggregate_slice_metrics, compute_slice_metrics

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".npz", ".nii", ".nii.gz"}
NEGATIVE_FOLDERS = {"good", "ungood_whole_patient_scans"}


def _parse_replacements(raw_items: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw_items:
        if ":" not in item:
            raise ValueError(f"Invalid replacement '{item}'. Expected 'src:dst'.")
        src, dst = item.split(":", 1)
        mapping[src] = dst
    return mapping


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
    stem_variants.extend([base, f"{base}_mask", f"{base}_label", f"{base}_gt"])

    alt_exts = set(VALID_EXTENSIONS)
    alt_exts.add(canonical_suffix(relative))
    alt_exts.add(relative.suffix.lower())
    alt_exts = {ext for ext in alt_exts if ext}

    candidates: list[Path] = []
    seen: set[Path] = set()
    for stem_variant in stem_variants:
        for ext in alt_exts:
            candidate = parent / f"{stem_variant}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _should_synthesize_negative_mask(relative: Path) -> bool:
    parts = {part.lower() for part in relative.parts}
    return bool(parts & NEGATIVE_FOLDERS)


def _resolve_ground_truth_path(
    ground_truth_root: Path,
    prediction_relative: Path,
    replacements: dict[str, str],
) -> tuple[Path | None, list[Path]]:
    replaced = apply_replacements(prediction_relative, replacements) if replacements else prediction_relative
    candidates_rel = _candidate_ground_truth_relatives(replaced)
    candidates: list[Path] = []
    root_name = ground_truth_root.name.lower()

    for rel in candidates_rel:
        candidate = ground_truth_root / rel
        candidates.append(candidate)
        if candidate.exists():
            return candidate, candidates

        # If the relative path already starts with the root name (e.g. “test/...”),
        # drop that component so we don’t get /…/test/test/...
        parts = rel.parts
        if parts and parts[0].lower() == root_name:
            alt = ground_truth_root / Path(*parts[1:])
            candidates.append(alt)
            if alt.exists():
                return alt, candidates

    return None, candidates



def _prepare_array(array: np.ndarray, threshold: float | None) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    data = project_to_2d(data)
    if threshold is not None:
        return (data >= threshold).astype(np.uint8)
    if data.dtype.kind in {"f", "i", "u"}:
        if data.max() <= 1.0:
            return (data > 0.5).astype(np.uint8)
        if data.max() <= 255:
            return (data > 0).astype(np.uint8)
    return (data > 0).astype(np.uint8)


@dataclass
class SliceRecord:
    relative_path: str
    ground_truth_path: str | None
    precision: float | None = None
    recall: float | None = None
    dice_score: float | None = None
    false_negative_rate: float | None = None
    balanced_accuracy: float | None = None
    true_positives: int | None = None
    false_positives: int | None = None
    true_negatives: int | None = None
    false_negatives: int | None = None
    missing_ground_truth: bool = False
    exclude_from_slice_metrics: bool = False


def compute_metrics(
    prediction_root: Path,
    ground_truth_root: Path,
    *,
    replacements: dict[str, str],
    prediction_threshold: float | None,
    ground_truth_threshold: float | None,
) -> tuple[list[SliceRecord], dict[str, float]]:
    prediction_files = [
        path
        for path in prediction_root.rglob("*")
        if path.is_file() and canonical_suffix(path) in VALID_EXTENSIONS
    ]
    if not prediction_files:
        raise FileNotFoundError(f"No prediction masks found in {prediction_root}")

    per_slice: list[SliceRecord] = []
    metrics_accumulator: list[dict[str, float]] = []

    for pred_path in prediction_files:
        relative = pred_path.relative_to(prediction_root)
        parts = {part.lower() for part in relative.parts}
        is_whole_patient = "ungood_whole_patient_scans" in parts
        is_true_anomaly = "ungood" in parts and not is_whole_patient

        record = SliceRecord(relative_path=str(relative), ground_truth_path=None)
        record.exclude_from_slice_metrics = is_whole_patient and not is_true_anomaly

        prediction_array = _prepare_array(load_array(pred_path).data, prediction_threshold)

        gt_path, candidates = _resolve_ground_truth_path(ground_truth_root, relative, replacements)
        if gt_path is None:
            if _should_synthesize_negative_mask(relative):
                ground_truth_array = np.zeros_like(prediction_array, dtype=np.uint8)
                record.ground_truth_path = "__synthetic_zero_mask__"
            else:
                record.missing_ground_truth = True
                per_slice.append(record)
                continue
        else:
            ground_truth_array = _prepare_array(load_array(gt_path).data, ground_truth_threshold)
            record.ground_truth_path = str(gt_path.relative_to(ground_truth_root))

        if prediction_array.shape != ground_truth_array.shape:
            raise ValueError(
                f"Shape mismatch for {relative}: prediction {prediction_array.shape}, "
                f"ground truth {ground_truth_array.shape}"
            )

        metrics = compute_slice_metrics(
            prediction_array,
            ground_truth_array,
            prediction_threshold=None,
            ground_truth_threshold=None,
        )
        if not record.exclude_from_slice_metrics:
            metrics_accumulator.append(metrics)

        record.precision = metrics["precision"]
        record.recall = metrics["recall"]
        record.dice_score = metrics["f1_score"]
        record.false_negative_rate = metrics["false_negative_rate"]
        record.balanced_accuracy = metrics["balanced_accuracy"]
        record.true_positives = int(metrics["true_positives"])
        record.false_positives = int(metrics["false_positives"])
        record.true_negatives = int(metrics["true_negatives"])
        record.false_negatives = int(metrics["false_negatives"])

        per_slice.append(record)

    aggregated = aggregate_slice_metrics(metrics_accumulator)
    aggregated["dice_score"] = aggregated.get("f1_score", None)
    return per_slice, aggregated


def _write_json(path: Path, per_slice: list[SliceRecord], aggregated: dict[str, float]) -> None:
    payload: dict[str, Any] = {
        "aggregated": aggregated,
        "slices": [asdict(record) for record in per_slice],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_csv(path: Path, per_slice: list[SliceRecord]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(per_slice[0]).keys()) if per_slice else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in per_slice:
            writer.writerow(asdict(record))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pixel-level segmentation metrics.")
    parser.add_argument("--prediction-dir", type=Path, required=True, help="Directory with prediction masks.")
    parser.add_argument("--ground-truth-dir", type=Path, required=True, help="Directory with ground-truth masks.")
    parser.add_argument(
        "--ground-truth-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Path component replacements applied when resolving ground-truth masks (default adds 'img:label').",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=None,
        help="Optional threshold applied before binarising predictions (default infers from data).",
    )
    parser.add_argument(
        "--ground-truth-threshold",
        type=float,
        default=None,
        help="Optional threshold applied before binarising ground truth (default infers from data).",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON file to store metrics.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV file with per-slice metrics.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    replacements = _parse_replacements(args.ground_truth_replace or [])
    if "img" not in replacements:
        replacements = {"img": "label", **replacements}

    per_slice, aggregated = compute_metrics(
        args.prediction_dir.resolve(),
        args.ground_truth_dir.resolve(),
        replacements=replacements,
        prediction_threshold=args.prediction_threshold,
        ground_truth_threshold=args.ground_truth_threshold,
    )

    dice = aggregated.get("dice_score")
    summary_lines = [
        f"Processed slices: {len([r for r in per_slice if not r.missing_ground_truth])}",
        f"Missing ground truth: {len([r for r in per_slice if r.missing_ground_truth])}",
        f"Precision: {aggregated['precision']:.4f}" if aggregated.get("precision") is not None else "Precision: N/A",
        f"Recall: {aggregated['recall']:.4f}" if aggregated.get("recall") is not None else "Recall: N/A",
        f"Dice score: {dice:.4f}" if dice is not None else "Dice score: N/A",
        f"False negative rate: {aggregated['false_negative_rate']:.4f}"
        if aggregated.get("false_negative_rate") is not None
        else "False negative rate: N/A",
        f"Balanced accuracy: {aggregated['balanced_accuracy']:.4f}"
        if aggregated.get("balanced_accuracy") is not None
        else "Balanced accuracy: N/A",
    ]
    print("\n".join(summary_lines))

    if args.output_json is not None:
        _write_json(args.output_json, per_slice, aggregated)
        print(f"[INFO] Metrics saved to {args.output_json}")

    if args.output_csv is not None:
        _write_csv(args.output_csv, per_slice)
        print(f"[INFO] Per-slice metrics saved to {args.output_csv}")


if __name__ == "__main__":
    main()
