"""Standardized binary-classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Calculate the standardized Phase 4 binary metrics."""
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    scores = np.asarray(y_score)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    has_both_classes = np.unique(truth).size == 2

    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions))
        if has_both_classes
        else None,
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(truth, scores))
        if has_both_classes
        else None,
        "roc_auc": float(roc_auc_score(truth, scores)) if has_both_classes else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support_negative": int((truth == 0).sum()),
        "support_positive": int((truth == 1).sum()),
    }
