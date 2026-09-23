"""Tests for standardized baseline evaluation."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import binary_classification_metrics


def test_binary_metrics_include_expected_confusion_counts() -> None:
    metrics = binary_classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.4, 0.9]),
    )

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_auc_metrics_are_none_when_split_has_one_class() -> None:
    metrics = binary_classification_metrics(
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0.1, 0.2]),
    )

    assert metrics["pr_auc"] is None
    assert metrics["roc_auc"] is None
    assert metrics["balanced_accuracy"] is None
