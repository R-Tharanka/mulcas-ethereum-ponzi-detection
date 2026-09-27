"""Protocol tests for Phase 6D1."""

from __future__ import annotations

import json

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, issparse

from src.experiments.phase6a import load_development_data, verify_frozen_manifest
from src.experiments.phase6b import baseline_parameters
from src.experiments.phase6d1 import DEFAULT_CONFIG_PATH, build_representation_pair, load_config
from src.experiments.temporal_cv import make_positive_occurrence_folds


def test_temporal_folds_reused_and_final_test_excluded() -> None:
    config = load_config(DEFAULT_CONFIG_PATH); partitions = load_development_data(config["dataset"], config["split_membership"])
    assert set(partitions) == {"train", "validation"}
    folds = make_positive_occurrence_folds(partitions["train"], config["temporal_cv"]["train_positive_ends"], config["temporal_cv"]["validation_positive_ends"])
    prior = json.loads((config["output_dir"].parent / "phase6c/temporal_cv_folds.json").read_text())
    indexed = partitions["train"].set_index("row_index")
    for fold, expected in zip(folds, prior, strict=True):
        train = indexed.loc[list(fold.train_indices)]; later = indexed.loc[list(fold.validation_indices)]
        assert train.createdBlockNumber.max() < later.createdBlockNumber.min()
        assert len(train) == expected["train_rows"] and len(later) == expected["validation_rows"]


def test_scaler_is_training_only_and_sparse_is_preserved() -> None:
    train = pd.DataFrame({"opcode":["ADD","ADD ADD"], "creator":["a","b"], "createdBlockNumber":[1,2], "label":[0,1]})
    later = pd.DataFrame({"opcode":["ADD ADD ADD"], "creator":["c"], "createdBlockNumber":[3]})
    baseline = (csr_matrix([[1],[2]]), csr_matrix([[3]])); structural = (csr_matrix(np.ones((2, 74))), csr_matrix(np.ones((1, 74)) * 100)); creator = (csr_matrix([[0],[0]]), csr_matrix([[0]]))
    train_x, later_x, audit = build_representation_pair({"baseline":True,"groups":["A"]}, "ridge", train, later, baseline, structural, creator)
    assert issparse(train_x) and issparse(later_x)
    assert audit["scaler_fit_rows"] == 2


def test_frozen_models_thresholds_and_all_manifests() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert baseline_parameters("ridge", {"class_weight":None})["class_weight"] is None
    assert baseline_parameters("svm", {"class_weight":None})["class_weight"] is None
    assert baseline_parameters("xgboost", {})["scale_pos_weight"] == 1.0
    assert len(verify_frozen_manifest(config["frozen_phase55_manifest"])) == 12
    assert len(verify_frozen_manifest(config["frozen_phase6a_manifest"])) == 7
    assert len(verify_frozen_manifest(config["frozen_phase6b_manifest"])) == 9
    assert len(verify_frozen_manifest(config["frozen_phase6c_manifest"])) == 8
