"""Safety and determinism tests for Phase 6A."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.phase6a import (
    DEFAULT_CONFIG_PATH,
    candidate_parameters,
    evaluate_once,
    load_config,
    load_development_data,
    select_candidate,
    verify_frozen_manifest,
)
from src.experiments.temporal_cv import make_positive_occurrence_folds


def test_temporal_cv_never_trains_on_future_contracts() -> None:
    frame = pd.DataFrame(
        {
            "row_index": range(8),
            "createdBlockNumber": [10, 20, 30, 40, 50, 60, 70, 80],
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    folds = make_positive_occurrence_folds(frame, [1, 2, 3], [2, 3, 4])
    indexed = frame.set_index("row_index")
    for fold in folds:
        train = indexed.loc[list(fold.train_indices)]
        validation = indexed.loc[list(fold.validation_indices)]
        assert train.createdBlockNumber.max() < validation.createdBlockNumber.min()
        assert not set(fold.train_indices) & set(fold.validation_indices)


def test_phase6a_development_loader_excludes_final_test() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    partitions = load_development_data(config["dataset"], config["split_membership"])
    assert set(partitions) == {"train", "validation"}
    assert len(partitions["train"]) == 5839
    assert len(partitions["validation"]) == 150
    assert 509 not in {len(frame) for frame in partitions.values()}


def test_frozen_phase55_manifest_is_unchanged() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    observed = verify_frozen_manifest(config["frozen_manifest"])
    assert len(observed) == 12


def test_cost_parameters_are_applied_to_training_model_only() -> None:
    ridge, class_weight, scale = candidate_parameters(
        "ridge", {"alpha": 1.0, "class_weight": None}, {"positive_weight": 5.0}
    )
    xgboost, xgb_class_weight, xgb_scale = candidate_parameters(
        "xgboost", {"n_estimators": 10}, {"scale_pos_weight": 28.195}
    )
    assert ridge["class_weight"] == {0: 1.0, 1: 5.0}
    assert class_weight == {0: 1.0, 1: 5.0}
    assert scale is None
    assert xgboost["scale_pos_weight"] == 28.195
    assert xgb_class_weight is None
    assert xgb_scale == 28.195


def test_seeded_experiment_metrics_are_deterministic() -> None:
    train_x = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    train_y = np.asarray([0, 0, 0, 1, 1, 1])
    validation_x = np.asarray([[0.5], [2.5], [4.5]])
    validation_y = np.asarray([0, 1, 1])
    parameters = {"alpha": 1.0, "class_weight": {0: 1.0, 1: 5.0}}
    first, _ = evaluate_once(
        "ridge", parameters, 42, train_x, train_y, validation_x, validation_y
    )
    second, _ = evaluate_once(
        "ridge", parameters, 42, train_x, train_y, validation_x, validation_y
    )
    assert first == second


def test_candidate_selection_uses_cv_metrics_and_precision_constraint() -> None:
    records = [
        {
            "experiment_id": "low-precision",
            "cv_precision_mean": 0.70,
            "cv_recall_mean": 1.0,
            "cv_f1_mean": 0.90,
            "cv_pr_auc_mean": 0.90,
        },
        {
            "experiment_id": "eligible",
            "cv_precision_mean": 0.85,
            "cv_recall_mean": 0.70,
            "cv_f1_mean": 0.76,
            "cv_pr_auc_mean": 0.80,
        },
    ]
    selected, fallback = select_candidate(records, minimum_precision=0.80)
    assert selected["experiment_id"] == "eligible"
    assert fallback is False
