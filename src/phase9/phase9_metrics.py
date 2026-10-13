"""Frozen Phase 9B pooled metrics and paired uncertainty."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_curve

from src.evaluation.metrics import binary_classification_metrics


def prevalence_standardized_precision(tpr: float, fpr: float, prevalence: float) -> float:
    denominator = tpr * prevalence + fpr * (1.0 - prevalence)
    return float((tpr * prevalence) / denominator) if denominator else 0.0


def recall_at_fpr_ceiling(y_true: np.ndarray, y_score: np.ndarray, ceiling: float) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    fpr, tpr, thresholds = roc_curve(truth, scores, drop_intermediate=False)
    allowed = np.flatnonzero(fpr <= ceiling + 1e-15)
    best_recall = float(tpr[allowed].max())
    tied = allowed[np.isclose(tpr[allowed], best_recall)]
    selected = int(tied[np.argmin(fpr[tied])])
    return {
        "recall_at_reference_fpr_ceiling": best_recall,
        "ranking_fpr": float(fpr[selected]),
        "ranking_score_threshold": float(thresholds[selected]),
    }


def pooled_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    fpr_ceiling: float,
    reference_prevalence: float,
) -> dict[str, Any]:
    result = binary_classification_metrics(y_true, y_pred, y_score)
    result["fpr"] = float(result["fp"] / max(result["fp"] + result["tn"], 1))
    result["prevalence_standardized_precision"] = prevalence_standardized_precision(
        float(result["recall"]), result["fpr"], reference_prevalence
    )
    result.update(recall_at_fpr_ceiling(y_true, y_score, fpr_ceiling))
    return result


def paired_bootstrap(
    control: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    truth = np.asarray(control["truth"], dtype=int)
    if not np.array_equal(truth, np.asarray(candidate["truth"], dtype=int)):
        raise ValueError("Paired bootstrap requires identical truth ordering.")
    n = len(truth)
    rng = np.random.default_rng(seed)
    deltas = {"f1": [], "pr_auc": [], "recall": []}
    for _ in range(replicates):
        sample = rng.integers(0, n, size=n)
        sampled_truth = truth[sample]
        control_pred = np.asarray(control["prediction"], dtype=int)[sample]
        candidate_pred = np.asarray(candidate["prediction"], dtype=int)[sample]
        control_score = np.asarray(control["score"], dtype=float)[sample]
        candidate_score = np.asarray(candidate["score"], dtype=float)[sample]
        deltas["f1"].append(
            f1_score(sampled_truth, candidate_pred, zero_division=0)
            - f1_score(sampled_truth, control_pred, zero_division=0)
        )
        deltas["recall"].append(
            recall_score(sampled_truth, candidate_pred, zero_division=0)
            - recall_score(sampled_truth, control_pred, zero_division=0)
        )
        if np.unique(sampled_truth).size == 2:
            deltas["pr_auc"].append(
                average_precision_score(sampled_truth, candidate_score)
                - average_precision_score(sampled_truth, control_score)
            )
    output: dict[str, float | int] = {"replicates": replicates, "seed": seed}
    for metric, values in deltas.items():
        array = np.asarray(values, dtype=float)
        output[f"delta_{metric}_mean"] = float(array.mean())
        output[f"delta_{metric}_ci_low"] = float(np.quantile(array, 0.025))
        output[f"delta_{metric}_ci_high"] = float(np.quantile(array, 0.975))
    return output
