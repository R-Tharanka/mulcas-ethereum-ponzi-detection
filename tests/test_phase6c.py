"""Leakage, threshold semantics, and determinism tests for Phase 6C."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score

from src.experiments.phase6a import build_feature_pair, load_development_data, verify_frozen_manifest
from src.experiments.phase6b import baseline_parameters
from src.experiments.phase6c import (
    DEFAULT_CONFIG_PATH,
    candidate_thresholds,
    load_config,
    select_threshold,
    threshold_metrics,
)
from src.experiments.temporal_cv import make_positive_occurrence_folds


def test_phase6c_reuses_forward_only_folds_and_excludes_final_test() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    partitions = load_development_data(config["dataset"], config["split_membership"])
    assert set(partitions) == {"train", "validation"}
    folds = make_positive_occurrence_folds(
        partitions["train"], config["temporal_cv"]["train_positive_ends"],
        config["temporal_cv"]["validation_positive_ends"])
    prior = json.loads((config["output_dir"].parent / "phase6b/temporal_cv_folds.json").read_text())
    indexed = partitions["train"].set_index("row_index")
    for fold, expected in zip(folds, prior, strict=True):
        train = indexed.loc[list(fold.train_indices)]
        later = indexed.loc[list(fold.validation_indices)]
        assert train.createdBlockNumber.max() < later.createdBlockNumber.min()
        assert not set(fold.train_indices) & set(fold.validation_indices)
        assert len(train) == expected["train_rows"] and len(later) == expected["validation_rows"]


def test_feature_preprocessing_is_fold_local() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    training = load_development_data(config["dataset"], config["split_membership"])["train"]
    fold = make_positive_occurrence_folds(training, [50], [100])[0]
    indexed = training.set_index("row_index", drop=False)
    past = indexed.loc[list(fold.train_indices)].reset_index(drop=True)
    later = indexed.loc[list(fold.validation_indices)].reset_index(drop=True)
    past_x, later_x, audit = build_feature_pair("opcode_term_count", past, later, ngram_min_df=2)
    assert audit["feature_fit_rows"] == len(past)
    assert audit["feature_fit_block_max"] < audit["evaluation_block_min"]
    assert past_x.shape[1] == later_x.shape[1]


def test_frozen_baseline_is_unweighted_unsampled_and_defaults_are_native() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config["models"]["ridge"]["default_threshold"] == 0.0
    assert config["models"]["svm"]["default_threshold"] == 0.0
    assert config["models"]["xgboost"]["default_threshold"] == 0.5
    assert config["models"]["ridge"]["score_type"] == "decision_function"
    assert config["models"]["svm"]["score_type"] == "decision_function"
    assert config["models"]["xgboost"]["score_type"] == "probability"
    assert baseline_parameters("ridge", {"class_weight": None})["class_weight"] is None
    assert baseline_parameters("svm", {"class_weight": None})["class_weight"] is None
    assert baseline_parameters("xgboost", {})["scale_pos_weight"] == 1.0


def test_candidate_grid_always_includes_default_and_is_deterministic() -> None:
    scores = np.asarray([-2.0, -0.5, 0.1, 0.8, 3.0])
    first = candidate_thresholds(scores, 0.0, 11)
    second = candidate_thresholds(scores, 0.0, 11)
    assert 0.0 in first
    assert np.array_equal(first, second)
    assert len(first) <= 12


def test_threshold_selection_uses_only_supplied_oof_records() -> None:
    records = [
        {"threshold": -0.2, "pooled_cv_precision": 0.79, "pooled_cv_recall": 1.0, "pooled_cv_f1": 0.88, "pooled_cv_balanced_accuracy": 0.9},
        {"threshold": 0.1, "pooled_cv_precision": 0.8, "pooled_cv_recall": 0.7, "pooled_cv_f1": 0.75, "pooled_cv_balanced_accuracy": 0.8},
        {"threshold": 0.2, "pooled_cv_precision": 0.9, "pooled_cv_recall": 0.6, "pooled_cv_f1": 0.72, "pooled_cv_balanced_accuracy": 0.79},
    ]
    selected, fallback = select_threshold(records, minimum_precision=0.8, default=0.0)
    assert selected is records[1]
    assert fallback is records[0]


def test_pr_auc_is_invariant_to_threshold_for_identical_scores() -> None:
    labels = np.asarray([0, 0, 1, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.3, 0.7, 0.9])
    low = threshold_metrics(labels, scores, 0.2)
    high = threshold_metrics(labels, scores, 0.8)
    expected = average_precision_score(labels, scores)
    assert low["pr_auc"] == high["pr_auc"] == expected
    assert low["tp"] != high["tp"]


def test_prior_phase_manifests_and_mulcas_config_are_unchanged() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    phase55 = verify_frozen_manifest(config["frozen_phase55_manifest"])
    assert len(phase55) == 12
    assert len(verify_frozen_manifest(config["frozen_phase6a_manifest"])) == 7
    assert len(verify_frozen_manifest(config["frozen_phase6b_manifest"])) == 9
    assert phase55["configs/paper_aligned_mulcas.yaml"]
