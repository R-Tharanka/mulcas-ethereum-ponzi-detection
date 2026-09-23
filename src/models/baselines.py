"""Fixed-parameter baseline viewer families from the published study."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_baseline_model(
    family: str,
    parameters: dict[str, Any],
    seed: int,
) -> BaseEstimator:
    """Construct one deterministic baseline classifier without tuning."""
    if family == "ridge":
        return RidgeClassifier(**parameters)
    if family == "svm":
        return SVC(random_state=seed, **parameters)
    if family == "xgboost":
        return XGBClassifier(random_state=seed, **parameters)
    raise ValueError(f"Unknown baseline model family: {family}")


def positive_class_scores(model: BaseEstimator, features: Any) -> np.ndarray:
    """Return continuous positive-class scores for ranking metrics."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return np.asarray(probabilities)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features)).reshape(-1)
    raise TypeError("Model must expose predict_proba or decision_function.")
