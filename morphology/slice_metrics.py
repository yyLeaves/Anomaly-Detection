"""
Utility functions for per-slice segmentation metrics.

These helpers operate on binary (or binarisable) masks and provide the
requested statistics: precision, recall, F1 score, false negative rate,
and balanced accuracy. Counts are returned alongside the metrics so that
callers can aggregate results across multiple slices.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "compute_metrics_from_counts",
    "compute_slice_metrics",
    "aggregate_slice_metrics",
]


def _binarize(mask: np.ndarray, threshold: float | None) -> np.ndarray:
    """Convert arbitrary arrays to boolean masks."""
    array = np.asarray(mask)
    if threshold is None:
        return array.astype(bool)
    return array >= threshold


def _safe_division(numerator: float, denominator: float, default: float) -> float:
    """Return numerator / denominator with a fallback for zero denominators."""
    if denominator == 0:
        return default
    return numerator / denominator


def compute_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """Compute slice-level metrics from a confusion matrix."""
    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)

    pos_support = tp + fn
    neg_support = tn + fp

    precision_default = 1.0 if (tp + fp == 0 and pos_support == 0) else 0.0
    precision = _safe_division(tp, tp + fp, precision_default)

    recall = _safe_division(tp, pos_support, 1.0 if pos_support == 0 else 0.0)
    fnr = _safe_division(fn, pos_support, 0.0 if pos_support == 0 else 0.0)
    tnr = _safe_division(tn, neg_support, 1.0 if neg_support == 0 else 0.0)

    f1_denominator = (2 * tp) + fp + fn
    f1_score = _safe_division(2 * tp, f1_denominator, 0.0)

    balanced_accuracy = (recall + tnr) / 2.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "false_negative_rate": float(fnr),
        "balanced_accuracy": float(balanced_accuracy),
    }


def compute_slice_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    prediction_threshold: float | None = None,
    ground_truth_threshold: float | None = None,
) -> dict[str, float]:
    """
    Compute confusion counts and metrics for a single slice.

    Parameters
    ----------
    prediction:
        Array-like structure containing the predicted mask.
    ground_truth:
        Array-like structure containing the ground-truth mask.
    prediction_threshold:
        Optional threshold applied before binarising the prediction.
    ground_truth_threshold:
        Optional threshold applied before binarising the ground truth.

    Returns
    -------
    dict[str, float]
        Dictionary containing confusion counts and derived metrics.
    """
    pred_mask = _binarize(prediction, prediction_threshold)
    gt_mask = _binarize(ground_truth, ground_truth_threshold)

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Prediction and ground-truth shapes differ: {pred_mask.shape} vs {gt_mask.shape}"
        )

    pred_flat = pred_mask.ravel()
    gt_flat = gt_mask.ravel()

    tp = int(np.count_nonzero(pred_flat & gt_flat))
    fp = int(np.count_nonzero(pred_flat & ~gt_flat))
    tn = int(np.count_nonzero(~pred_flat & ~gt_flat))
    fn = int(np.count_nonzero(~pred_flat & gt_flat))

    metrics = compute_metrics_from_counts(tp, fp, tn, fn)
    metrics.update(
        {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "positive_support": tp + fn,
            "negative_support": tn + fp,
        }
    )
    return metrics


def aggregate_slice_metrics(results: Iterable[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-slice results into a single confusion matrix and metrics."""
    totals = {
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
    }
    slice_count = 0

    for result in results:
        slice_count += 1
        totals["true_positives"] += int(result.get("true_positives", 0))
        totals["false_positives"] += int(result.get("false_positives", 0))
        totals["true_negatives"] += int(result.get("true_negatives", 0))
        totals["false_negatives"] += int(result.get("false_negatives", 0))

    tp = totals["true_positives"]
    fp = totals["false_positives"]
    tn = totals["true_negatives"]
    fn = totals["false_negatives"]

    aggregated = compute_metrics_from_counts(tp, fp, tn, fn)
    aggregated.update(
        {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "positive_support": tp + fn,
            "negative_support": tn + fp,
            "num_slices": slice_count,
        }
    )
    return aggregated
