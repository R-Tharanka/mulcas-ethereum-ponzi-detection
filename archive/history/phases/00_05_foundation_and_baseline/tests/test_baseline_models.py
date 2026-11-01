"""Tests for fixed deterministic baseline model helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.baselines import build_baseline_model, positive_class_scores


@pytest.mark.parametrize("family", ["ridge", "svm", "xgboost"])
def test_baseline_family_fits_and_exposes_scores(family: str) -> None:
    parameters = {
        "ridge": {"alpha": 1.0},
        "svm": {"C": 1.0, "kernel": "linear"},
        "xgboost": {
            "n_estimators": 5,
            "max_depth": 2,
            "n_jobs": 1,
            "eval_metric": "logloss",
        },
    }[family]
    features = np.array([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    model = build_baseline_model(family, parameters, seed=42).fit(features, labels)

    assert model.predict(features).shape == (4,)
    assert positive_class_scores(model, features).shape == (4,)


def test_unknown_baseline_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown baseline"):
        build_baseline_model("unknown", {}, seed=42)
