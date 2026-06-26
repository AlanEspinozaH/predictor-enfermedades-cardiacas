"""Shared binary-classification metrics used by validation and fairness scripts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")
    return value


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float] | None:
    """Return a 95% Wilson interval for a binomial proportion."""

    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1 + (z**2 / trials)
    centre = proportion + (z**2 / (2 * trials))
    margin = z * math.sqrt(
        (proportion * (1 - proportion) / trials) + (z**2 / (4 * trials**2))
    )
    return [
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    ]


def binary_classification_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float,
) -> dict[str, Any]:
    """Calculate threshold-dependent and ranking metrics for a binary target."""

    selected_threshold = validate_threshold(threshold)
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)

    if truth.ndim != 1 or scores.ndim != 1 or len(truth) != len(scores):
        raise ValueError("y_true and y_score must be one-dimensional and aligned.")
    if len(truth) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("y_score must contain finite probabilities in [0, 1].")

    predicted = (scores >= selected_threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics: dict[str, Any] = {
        "threshold": selected_threshold,
        "row_count": int(len(truth)),
        "positive_count": int(truth.sum()),
        "prevalence": float(truth.mean()),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "recall_95_ci": wilson_interval(tp, tp + fn),
    }
    if len(np.unique(truth)) == 2:
        metrics.update(
            {
                "roc_auc": float(roc_auc_score(truth, scores)),
                "pr_auc": float(average_precision_score(truth, scores)),
                "brier_score": float(brier_score_loss(truth, scores)),
            }
        )
    return metrics
