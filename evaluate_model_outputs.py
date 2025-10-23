#!/usr/bin/env python3
"""Compute pixel-level, slice-level, and patient-level metrics for segmentation predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from compute_pixel_metrics import SliceRecord, _parse_replacements, compute_metrics


def _safe_divide(numerator: float, denominator: float, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _compute_image_level_metrics(per_slice: list[SliceRecord]) -> dict[str, float]:
    tp = fp = tn = fn = 0
    missing = 0

    for record in per_slice:
        if record.missing_ground_truth:
            missing += 1
            continue
        pred_positive = (record.true_positives or 0) + (record.false_positives or 0) > 0
        gt_positive = (record.true_positives or 0) + (record.false_negatives or 0) > 0

        if pred_positive and gt_positive:
            tp += 1
        elif pred_positive and not gt_positive:
            fp += 1
        elif not pred_positive and gt_positive:
            fn += 1
        else:
            tn += 1

    precision = _safe_divide(tp, tp + fp, 0.0)
    recall = _safe_divide(tp, tp + fn, 0.0)
    dice = _safe_divide(2 * tp, 2 * tp + fp + fn, 0.0)
    fnr = _safe_divide(fn, tp + fn, 0.0)
    tnr = _safe_divide(tn, tn + fp, 0.0)
    balanced_accuracy = (recall + tnr) / 2.0

    return {
        "precision": precision,
        "recall": recall,
        "dice_score": dice,
        "false_negative_rate": fnr,
        "balanced_accuracy": balanced_accuracy,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "evaluated_slices": tp + fp + tn + fn,
        "missing_ground_truth_slices": missing,
    }


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _parse_patient_id(relative_path: str) -> str:
    stem = Path(relative_path).stem
    if stem.endswith("_pred_mask"):
        stem = stem[: -len("_pred_mask")]
    return stem.rsplit("_", 1)[0]


def _compute_mean_positive_fraction_metrics(
    per_slice: list[SliceRecord],
    thresholds: list[float],
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    patient_fractions: dict[str, list[float]] = defaultdict(list)
    patient_gt_positive: dict[str, bool] = defaultdict(bool)

    for record in per_slice:
        if record.missing_ground_truth:
            continue

        total_pixels = (
            (record.true_positives or 0)
            + (record.false_positives or 0)
            + (record.true_negatives or 0)
            + (record.false_negatives or 0)
        )
        if total_pixels == 0:
            continue

        predicted_positive = (record.true_positives or 0) + (record.false_positives or 0)
        fraction = predicted_positive / total_pixels

        patient_id = _parse_patient_id(record.relative_path)
        patient_fractions[patient_id].append(fraction)

        gt_positive_pixels = (record.true_positives or 0) + (record.false_negatives or 0)
        if gt_positive_pixels > 0:
            patient_gt_positive[patient_id] = True

    patient_mean_fraction: dict[str, float] = {}
    for patient_id, fractions in patient_fractions.items():
        if fractions:
            patient_mean_fraction[patient_id] = sum(fractions) / len(fractions)

    summary = {
        "num_patients": len(patient_mean_fraction),
        "num_patients_gt_positive": sum(1 for value in patient_gt_positive.values() if value),
        "mean_fraction_by_patient": patient_mean_fraction,
    }

    metrics_per_threshold: list[dict[str, float]] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for patient_id, mean_fraction in patient_mean_fraction.items():
            pred_positive = mean_fraction >= threshold
            gt_positive = patient_gt_positive.get(patient_id, False)

            if pred_positive and gt_positive:
                tp += 1
            elif pred_positive and not gt_positive:
                fp += 1
            elif not pred_positive and gt_positive:
                fn += 1
            else:
                tn += 1

        precision = _safe_divide(tp, tp + fp, 0.0)
        recall = _safe_divide(tp, tp + fn, 0.0)
        dice = _safe_divide(2 * tp, 2 * tp + fp + fn, 0.0)
        fnr = _safe_divide(fn, tp + fn, 0.0)
        tnr = _safe_divide(tn, tn + fp, 0.0)
        balanced_accuracy = (recall + tnr) / 2.0

        metrics_per_threshold.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "dice_score": dice,
                "false_negative_rate": fnr,
                "balanced_accuracy": balanced_accuracy,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            }
        )

    return metrics_per_threshold, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute image-level and pixel-level metrics for prediction masks."
    )
    parser.add_argument("--prediction-dir", type=Path, required=True, help="Directory with prediction masks.")
    parser.add_argument("--ground-truth-dir", type=Path, required=True, help="Directory with ground-truth masks.")
    parser.add_argument(
        "--ground-truth-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Path component replacements applied before looking for ground-truth masks (default adds 'img:label').",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=None,
        help="Optional threshold applied before binarising predictions.",
    )
    parser.add_argument(
        "--ground-truth-threshold",
        type=float,
        default=None,
        help="Optional threshold applied before binarising ground truth.",
    )
    parser.add_argument(
        "--mean-fraction-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1],
        help="Thresholds (alpha_mean) for patient-level classification via mean positive fraction.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file. Stores metrics and per-slice outputs.",
    )
    return parser.parse_args(argv)


def evaluate_outputs(
    prediction_dir: Path,
    ground_truth_dir: Path,
    *,
    ground_truth_replacements: dict[str, str],
    prediction_threshold: float | None,
    ground_truth_threshold: float | None,
    mean_fraction_thresholds: list[float],
    print_summary: bool = True,
) -> dict[str, Any]:
    per_slice, pixel_metrics = compute_metrics(
        prediction_dir.resolve(),
        ground_truth_dir.resolve(),
        replacements=ground_truth_replacements,
        prediction_threshold=prediction_threshold,
        ground_truth_threshold=ground_truth_threshold,
    )

    image_metrics = _compute_image_level_metrics(per_slice)
    mean_fraction_metrics, patient_summary = _compute_mean_positive_fraction_metrics(per_slice, mean_fraction_thresholds)

    if print_summary:
        print("Pixel-level metrics:")
        print(f"  Precision:           {_format_metric(pixel_metrics.get('precision'))}")
        print(f"  Recall:              {_format_metric(pixel_metrics.get('recall'))}")
        dice_metric = pixel_metrics.get("dice_score", pixel_metrics.get("f1_score"))
        print(f"  Dice score:          {_format_metric(dice_metric)}")
        print(f"  False negative rate: {_format_metric(pixel_metrics.get('false_negative_rate'))}")
        print(f"  Balanced accuracy:   {_format_metric(pixel_metrics.get('balanced_accuracy'))}")
        print()
        print("Slice-level image metrics (predicted positives vs. ground-truth positives):")
        print(f"  Precision:           {_format_metric(image_metrics['precision'])}")
        print(f"  Recall:              {_format_metric(image_metrics['recall'])}")
        print(f"  Dice score:          {_format_metric(image_metrics['dice_score'])}")
        print(f"  False negative rate: {_format_metric(image_metrics['false_negative_rate'])}")
        print(f"  Balanced accuracy:   {_format_metric(image_metrics['balanced_accuracy'])}")
        print(f"  Evaluated slices:    {image_metrics['evaluated_slices']}")
        print(f"  Missing ground truth:{image_metrics['missing_ground_truth_slices']}")
        print()
        print("Patient-level metrics (mean positive fraction):")
        print(f"  Patients evaluated:  {patient_summary['num_patients']}")
        print(f"  GT-positive patients:{patient_summary['num_patients_gt_positive']}")
        for metrics in mean_fraction_metrics:
            print(
                f"  α_mean={metrics['threshold']:.4f} → "
                f"P={_format_metric(metrics['precision'])}, "
                f"R={_format_metric(metrics['recall'])}, "
                f"Dice={_format_metric(metrics['dice_score'])}, "
                f"FNR={_format_metric(metrics['false_negative_rate'])}, "
                f"BA={_format_metric(metrics['balanced_accuracy'])}"
            )

    return {
        "pixel_metrics": pixel_metrics,
        "slice_image_metrics": image_metrics,
        "patient_mean_fraction_metrics": mean_fraction_metrics,
        "patient_summary": patient_summary,
        "per_slice": per_slice,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    replacements = _parse_replacements(args.ground_truth_replace or [])
    if "img" not in replacements:
        replacements = {"img": "label", **replacements}

    results = evaluate_outputs(
        args.prediction_dir,
        args.ground_truth_dir,
        ground_truth_replacements=replacements,
        prediction_threshold=args.prediction_threshold,
        ground_truth_threshold=args.ground_truth_threshold,
        mean_fraction_thresholds=args.mean_fraction_thresholds,
        print_summary=True,
    )

    if args.output_json is not None:
        payload = {
            "pixel_metrics": results["pixel_metrics"],
            "slice_image_metrics": results["slice_image_metrics"],
            "patient_mean_fraction_metrics": results["patient_mean_fraction_metrics"],
            "patient_summary": results["patient_summary"],
            "per_slice": [asdict(record) for record in results["per_slice"]],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[INFO] Metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
