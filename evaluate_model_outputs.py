#!/usr/bin/env python3
"""Compute pixel-level, slice-level, and patient-level metrics for segmentation predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from compute_pixel_metrics import SliceRecord, _parse_replacements, compute_metrics


def _safe_divide(numerator: float, denominator: float, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _compute_image_level_metrics(per_slice: list[SliceRecord]) -> dict[str, float]:
    tp = fp = tn = fn = 0
    missing = 0
    evaluated = 0

    for record in per_slice:
        if getattr(record, "exclude_from_slice_metrics", False):
            continue
        if record.missing_ground_truth:
            missing += 1
            continue
        pred_positive = (record.true_positives or 0) + (record.false_positives or 0) > 0
        # Determine patient-level GT positivity from mask values rather than folder names
        gt_positive = (record.true_positives or 0) + (record.false_negatives or 0) > 0

        if pred_positive and gt_positive:
            tp += 1
        elif pred_positive and not gt_positive:
            fp += 1
        elif not pred_positive and gt_positive:
            fn += 1
        else:
            tn += 1
        evaluated += 1

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
        "evaluated_slices": evaluated,
        "missing_ground_truth_slices": missing,
    }


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return ROC-AUC if both positive and negative labels are present, otherwise None."""
    unique = np.unique(labels)
    if unique.size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _parse_patient_id(relative_path: str) -> str:
    stem = Path(relative_path).stem
    if stem.endswith("_pred_mask"):
        stem = stem[: -len("_pred_mask")]
    return stem.rsplit("_", 1)[0]


def _compute_mean_positive_fraction_metrics(
    per_slice: list[SliceRecord],
    thresholds: list[float],
) -> tuple[list[dict[str, float]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
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

        gt_positive = (record.true_positives or 0) + (record.false_negatives or 0) > 0
        if gt_positive:
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

    patient_threshold_details: dict[str, list[dict[str, Any]]] = {}
    for patient_id, mean_fraction in patient_mean_fraction.items():
        gt_positive = patient_gt_positive.get(patient_id, False)
        per_threshold: list[dict[str, Any]] = []
        for threshold in thresholds:
            pred_positive = mean_fraction >= threshold
            tpr = None
            if gt_positive:
                tpr = 1.0 if pred_positive else 0.0
            per_threshold.append(
                {
                    "threshold": threshold,
                    "mean_positive_fraction": mean_fraction,
                    "predicted_positive": pred_positive,
                    "ground_truth_positive": gt_positive,
                    "true_positive_rate": tpr,
                }
            )
        patient_threshold_details[patient_id] = per_threshold

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
                "true_positive_rate": recall,
                "dice_score": dice,
                "false_negative_rate": fnr,
                "balanced_accuracy": balanced_accuracy,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            }
        )

    return metrics_per_threshold, summary, patient_threshold_details


def compute_anomaly_map_aurocs(
    anomaly_map_root: Path,
    ground_truth_root: Path,
    *,
    replacements: dict[str, str],
    ground_truth_threshold: float | None,
) -> dict[str, Any]:
    """Compute pixel-level and image-level AUROCs for raw anomaly maps (pre post-processing)."""
    from compute_pixel_metrics import _resolve_ground_truth_path, _prepare_array, VALID_EXTENSIONS
    from fastflow_postprocess import canonical_suffix, load_array, project_to_2d

    anomaly_files = [
        path
        for path in anomaly_map_root.rglob("*")
        if path.is_file() and canonical_suffix(path) in VALID_EXTENSIONS
    ]
    if not anomaly_files:
        raise FileNotFoundError(f"No anomaly maps found in {anomaly_map_root}")

    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    image_scores: list[float] = []
    image_labels: list[int] = []
    missing_ground_truth = 0

    for anomaly_path in anomaly_files:
        relative = anomaly_path.relative_to(anomaly_map_root)
        gt_path, _ = _resolve_ground_truth_path(ground_truth_root, relative, replacements)
        if gt_path is None:
            missing_ground_truth += 1
            continue

        anomaly_array = project_to_2d(load_array(anomaly_path).data.astype(np.float32, copy=False))
        anomaly_array = np.nan_to_num(anomaly_array, nan=0.0, posinf=0.0, neginf=0.0)
        gt_array = _prepare_array(load_array(gt_path).data, ground_truth_threshold)

        if anomaly_array.shape != gt_array.shape:
            raise ValueError(
                f"Shape mismatch for {relative}: anomaly map {anomaly_array.shape}, ground truth {gt_array.shape}"
            )

        pixel_scores.append(anomaly_array.reshape(-1).astype(np.float32, copy=False))
        pixel_labels.append(gt_array.reshape(-1).astype(np.uint8, copy=False))
        image_scores.append(float(anomaly_array.max()))
        image_labels.append(1 if gt_array.max() > 0 else 0)

    result: dict[str, Any] = {
        "evaluated_slices": len(image_scores),
        "missing_ground_truth_slices": missing_ground_truth,
        "pixel_auroc": None,
        "image_auroc": None,
    }

    if pixel_scores:
        pixel_scores_arr = np.concatenate(pixel_scores)
        pixel_labels_arr = np.concatenate(pixel_labels)
        result["pixel_positive_fraction"] = float(pixel_labels_arr.mean()) if pixel_labels_arr.size else 0.0
        result["pixel_auroc"] = _safe_roc_auc(pixel_labels_arr, pixel_scores_arr)

    if image_scores:
        image_scores_arr = np.asarray(image_scores, dtype=np.float32)
        image_labels_arr = np.asarray(image_labels, dtype=np.uint8)
        result["image_positive_fraction"] = float(image_labels_arr.mean()) if image_labels_arr.size else 0.0
        result["image_auroc"] = _safe_roc_auc(image_labels_arr, image_scores_arr)

    return result


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
    parser.add_argument(
        "--anomaly-map-dir",
        type=Path,
        default=None,
        help="Optional directory with raw anomaly maps used to compute AUROC benchmarks.",
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
    anomaly_map_dir: Path | None = None,
) -> dict[str, Any]:
    per_slice, pixel_metrics = compute_metrics(
        prediction_dir.resolve(),
        ground_truth_dir.resolve(),
        replacements=ground_truth_replacements,
        prediction_threshold=prediction_threshold,
        ground_truth_threshold=ground_truth_threshold,
    )

    image_metrics = _compute_image_level_metrics(per_slice)
    mean_fraction_metrics, patient_summary, patient_threshold_details = _compute_mean_positive_fraction_metrics(
        per_slice, mean_fraction_thresholds
    )

    anomaly_auroc_metrics: dict[str, Any] | None = None
    if anomaly_map_dir is not None:
        try:
            anomaly_auroc_metrics = compute_anomaly_map_aurocs(
                anomaly_map_dir,
                ground_truth_dir,
                replacements=ground_truth_replacements,
                ground_truth_threshold=ground_truth_threshold,
            )
        except FileNotFoundError as exc:
            print(f"[WARN] Skipping anomaly-map AUROC computation: {exc}")

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
                f"TPR={_format_metric(metrics['true_positive_rate'])}, "
                f"Dice={_format_metric(metrics['dice_score'])}, "
                f"FNR={_format_metric(metrics['false_negative_rate'])}, "
                f"BA={_format_metric(metrics['balanced_accuracy'])}"
            )
        if patient_threshold_details:
            print()
            print("Per-patient true positive rates by threshold:")
            for patient_id in sorted(patient_threshold_details):
                entries = patient_threshold_details[patient_id]
                formatted = []
                for entry in entries:
                    tpr_value = entry["true_positive_rate"]
                    tpr_str = _format_metric(tpr_value)
                    formatted.append(f"α={entry['threshold']:.4f}:{tpr_str}")
                print(f"  {patient_id}: {' | '.join(formatted)}")

        if anomaly_auroc_metrics is not None:
            print()
            print("Raw anomaly-map AUROC benchmarks (pre post-processing):")
            print(f"  Pixel AUROC:         {_format_metric(anomaly_auroc_metrics.get('pixel_auroc'))}")
            print(f"  Image AUROC:         {_format_metric(anomaly_auroc_metrics.get('image_auroc'))}")
            print(
                f"  Evaluated slices:    {anomaly_auroc_metrics.get('evaluated_slices', 0)} "
                f"(missing={anomaly_auroc_metrics.get('missing_ground_truth_slices', 0)})"
            )

    return {
        "pixel_metrics": pixel_metrics,
        "slice_image_metrics": image_metrics,
        "patient_mean_fraction_metrics": mean_fraction_metrics,
        "patient_summary": patient_summary,
        "patient_threshold_details": patient_threshold_details,
        "per_slice": per_slice,
        "anomaly_auroc_metrics": anomaly_auroc_metrics,
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
        anomaly_map_dir=args.anomaly_map_dir,
    )

    if args.output_json is not None:
        payload = {
            "pixel_metrics": results["pixel_metrics"],
            "slice_image_metrics": results["slice_image_metrics"],
            "patient_mean_fraction_metrics": results["patient_mean_fraction_metrics"],
            "patient_summary": results["patient_summary"],
            "patient_threshold_details": results["patient_threshold_details"],
            "per_slice": [asdict(record) for record in results["per_slice"]],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[INFO] Metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
