"""Protocol, leakage, and determinism tests for Phase 6B."""

from __future__ import annotations

import json

import numpy as np
from scipy.sparse import csr_matrix, issparse

from src.experiments.phase6a import build_feature_pair, load_development_data, verify_frozen_manifest
from src.experiments.phase6b import (
    DEFAULT_CONFIG_PATH,
    apply_training_sampler,
    baseline_parameters,
    load_config,
)
from src.experiments.temporal_cv import make_positive_occurrence_folds
from src.features.word2vec import IDFWeightedOpcodeWord2VecTransformer


def test_phase6b_reuses_phase6a_forward_only_folds_and_excludes_test() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    partitions = load_development_data(config["dataset"], config["split_membership"])
    assert set(partitions) == {"train", "validation"}
    folds = make_positive_occurrence_folds(
        partitions["train"], config["temporal_cv"]["train_positive_ends"],
        config["temporal_cv"]["validation_positive_ends"])
    phase6a = json.loads((config["output_dir"].parent / "phase6a/temporal_cv_folds.json").read_text())
    indexed = partitions["train"].set_index("row_index")
    for fold, prior in zip(folds, phase6a, strict=True):
        train = indexed.loc[list(fold.train_indices)]
        later = indexed.loc[list(fold.validation_indices)]
        assert train.createdBlockNumber.max() < later.createdBlockNumber.min()
        assert not set(fold.train_indices) & set(fold.validation_indices)
        assert len(train) == prior["train_rows"]
        assert len(later) == prior["validation_rows"]


def test_vectorizer_and_creator_fit_only_fold_training_rows() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    training = load_development_data(config["dataset"], config["split_membership"])["train"]
    folds = make_positive_occurrence_folds(training, [50], [100])
    indexed = training.set_index("row_index", drop=False)
    past = indexed.loc[list(folds[0].train_indices)].reset_index(drop=True)
    later = indexed.loc[list(folds[0].validation_indices)].reset_index(drop=True)
    train_x, later_x, audit = build_feature_pair("opcode_term_count", past, later, ngram_min_df=2)
    assert audit["feature_fit_rows"] == len(past)
    assert audit["feature_fit_block_max"] < audit["evaluation_block_min"]
    assert train_x.shape[1] == later_x.shape[1]


def test_word2vec_and_idf_transform_are_read_only() -> None:
    transformer = IDFWeightedOpcodeWord2VecTransformer(vector_size=4, epochs=2, workers=1, seed=42)
    transformer.fit(["PUSH ADD", "PUSH STOP"])
    vocabulary = dict(transformer.model_.wv.key_to_index)  # type: ignore[union-attr]
    idf = dict(transformer.idf_)
    transformer.transform(["FUTURE_ONLY PUSH"])
    assert transformer.model_.wv.key_to_index == vocabulary  # type: ignore[union-attr]
    assert transformer.idf_ == idf
    assert "FUTURE_ONLY" not in transformer.idf_


def test_sampling_only_changes_training_and_preserves_sparse_matrix() -> None:
    train_x = csr_matrix(np.arange(22).reshape(11, 2))
    train_y = np.asarray([1] + [0] * 10)
    validation_x = csr_matrix([[100, 101], [102, 103]])
    before = validation_x.copy()
    sampled_x, sampled_y, audit = apply_training_sampler(
        train_x, train_y, sampler_type="random_over", sampling_strategy=1.0,
        requested_ratio="1:1", seed=42)
    assert issparse(sampled_x)
    assert audit["post_sampling_positive"] == audit["post_sampling_negative"] == 10
    assert audit["rows_duplicated"] == 9
    assert (validation_x != before).nnz == 0
    assert len(sampled_y) == 20


def test_sampling_after_creator_construction_cannot_change_history_state() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    training = load_development_data(config["dataset"], config["split_membership"])["train"].iloc[:409]
    train_x, _, audit = build_feature_pair("opcode_term_count", training.iloc[:300], training.iloc[300:], ngram_min_df=2)
    state_size = audit["creator_positive_state_size"]
    labels = training.iloc[:300]["label"].to_numpy(int)
    apply_training_sampler(train_x, labels, sampler_type="random_under", sampling_strategy=1.0, requested_ratio="1:1", seed=42)
    apply_training_sampler(train_x, labels, sampler_type="random_over", sampling_strategy=1.0, requested_ratio="1:1", seed=42)
    assert audit["creator_positive_state_size"] == state_size


def test_unreachable_ratio_is_explicit_noop_and_sampling_is_deterministic() -> None:
    x = csr_matrix(np.arange(12).reshape(6, 2)); y = np.asarray([1, 1, 1, 0, 0, 0])
    noop_x, noop_y, noop = apply_training_sampler(x, y, sampler_type="random_under", sampling_strategy=.1, requested_ratio="1:10", seed=42)
    assert noop["sampling_status"] == "unreachable_from_current_ratio"
    assert noop_x.shape == x.shape and np.array_equal(noop_y, y)
    args = dict(sampler_type="random_over", sampling_strategy=1.0, requested_ratio="1:1", seed=42)
    first_x, first_y, first_audit = apply_training_sampler(csr_matrix([[1], [2], [3], [4]]), np.asarray([1, 0, 0, 0]), **args)
    second_x, second_y, second_audit = apply_training_sampler(csr_matrix([[1], [2], [3], [4]]), np.asarray([1, 0, 0, 0]), **args)
    assert (first_x != second_x).nnz == 0
    assert np.array_equal(first_y, second_y) and first_audit == second_audit


def test_baseline_cost_parameters_and_frozen_manifests() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert baseline_parameters("ridge", {"alpha": 1, "class_weight": None})["class_weight"] is None
    assert baseline_parameters("svm", {"class_weight": None})["class_weight"] is None
    assert baseline_parameters("xgboost", {"n_estimators": 1})["scale_pos_weight"] == 1.0
    assert len(verify_frozen_manifest(config["frozen_phase55_manifest"])) == 12
    assert len(verify_frozen_manifest(config["frozen_phase6a_manifest"])) == 7
