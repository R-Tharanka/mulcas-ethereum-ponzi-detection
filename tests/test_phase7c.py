"""Protocol tests for Phase 7C chronological creator-feature ablation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from src.evaluation.temporal import build_ponzi_occurrence_periods
from src.experiments.phase6a import verify_frozen_manifest
from src.experiments.phase7a import (
    _load_frozen_configs,
    load_config as load_phase7a_config,
    load_full_dataset,
)
from src.experiments.phase7c import (
    DEFAULT_CONFIG_PATH,
    build_creator_ablation_matrices,
    creator_diagnostics,
    load_config,
)


def _small_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "opcode": ["PUSH1 ADD", "CALL STOP", "PUSH1 SSTORE", "ADD STOP"],
            "creator": ["a", "b", "a", "c"],
            "createdBlockNumber": [1, 2, 3, 4],
            "label": [0, 1, 1, 0],
        }
    )
    target = pd.DataFrame(
        {
            "row_index": [4, 5, 6],
            "opcode": ["FUTURE_ONLY STOP", "CALL STOP", "ADD STOP"],
            "creator": ["b", "future", "a"],
            "createdBlockNumber": [5, 6, 7],
            "label": [1, 0, 1],
        }
    )
    return train, target


def test_phase7a_period_definitions_are_reused_exactly() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    phase7a_config = load_phase7a_config(config["phase7a_config"])
    phase7a = json.loads(config["phase7a_results"].read_text(encoding="utf-8"))
    frame = load_full_dataset(phase7a_config["dataset"])
    _, definitions = build_ponzi_occurrence_periods(
        frame,
        positive_ends=phase7a_config["periods"]["positive_ends"],
        expected_positives=phase7a_config["periods"]["expected_total_positives"],
    )
    assert definitions == phase7a["period_definitions"]


def test_creator_ablation_removes_exactly_one_column_without_placeholder() -> None:
    train, target = _small_frames()
    matrices, audit, creator = build_creator_ablation_matrices(
        train, target, ["opcode_term_count", "opcode_tfidf"], ngram_min_df=2
    )
    assert audit["creator_column_removed_not_replaced"] is True
    assert set(creator) == {0.0, 1.0}
    for view in matrices["with_creator"]:
        with_train, with_target = matrices["with_creator"][view]
        without_train, without_target = matrices["without_creator"][view]
        assert issparse(with_train) and issparse(without_train)
        assert with_train.shape[1] == without_train.shape[1] + 1
        assert with_target.shape[1] == without_target.shape[1] + 1
        assert np.array_equal(with_train[:, :-1].toarray(), without_train.toarray())
        assert np.array_equal(with_target[:, :-1].toarray(), without_target.toarray())


def test_future_labels_cannot_change_creator_or_static_features() -> None:
    train, target = _small_frames()
    first, first_audit, first_creator = build_creator_ablation_matrices(
        train, target, ["opcode_term_count"], ngram_min_df=2
    )
    changed_target = target.assign(label=1 - target["label"])
    second, second_audit, second_creator = build_creator_ablation_matrices(
        train, changed_target, ["opcode_term_count"], ngram_min_df=2
    )
    assert first_audit == second_audit
    assert np.array_equal(first_creator, second_creator)
    for condition in ("with_creator", "without_creator"):
        for left, right in zip(first[condition]["opcode_term_count"], second[condition]["opcode_term_count"], strict=True):
            assert np.array_equal(left.toarray(), right.toarray())


def test_creator_diagnostics_keep_historical_state_fixed_between_conditions() -> None:
    train, target = _small_frames()
    _, _, creator = build_creator_ablation_matrices(
        train, target, ["opcode_term_count"], ngram_min_df=2
    )
    base = pd.DataFrame(
        {
            "sample_index": target["row_index"],
            "prediction": [1, 0, 0],
            "probability": [0.9, 0.1, 0.2],
            "handled_by": "viewer",
        }
    )
    diagnostics = creator_diagnostics(
        train,
        target,
        creator,
        {"with_creator": base, "without_creator": base.assign(prediction=[0, 0, 0])},
    )
    assert diagnostics["previously_observed_creator_count"] == 2
    assert diagnostics["historical_ponzi_creator_count"] == 2
    assert diagnostics["true_ponzi_with_creator_feature_1"] == 2
    assert diagnostics["with_creator_tp_creator_feature_1"] == 1
    assert diagnostics["without_creator_fn_creator_feature_1"] == 2


def test_model_architecture_and_thresholds_are_not_retuned() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    _, models, mulcas = _load_frozen_configs(load_phase7a_config(config["phase7a_config"]))
    assert config["protocol"]["optimization"] == "prohibited"
    assert config["protocol"]["retuning"] == "prohibited"
    assert config["creator_ablation"] == {
        "operation": "remove_final_creator_column",
        "replacement_column": "none",
        "expected_dimension_difference": 1,
    }
    assert mulcas["complexity_control_n"] == 2
    assert mulcas["voting_threshold"] == 0
    assert [(item.get("negative_threshold"), item.get("positive_threshold")) for item in mulcas["viewer_sequence"][:2]] == [
        (0.05, 0.95), (0.05, 0.51)
    ]
    assert models["models"]["ridge"]["class_weight"] is None
    assert models["models"]["xgboost"].get("scale_pos_weight", 1) == 1


def test_phase7a_and_phase7b_artifacts_remain_frozen() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert len(verify_frozen_manifest(config["frozen_manifest"])) == 10


def test_matrix_construction_is_deterministic() -> None:
    train, target = _small_frames()
    first, first_audit, first_creator = build_creator_ablation_matrices(
        train, target, ["opcode_term_count"], ngram_min_df=2
    )
    second, second_audit, second_creator = build_creator_ablation_matrices(
        train, target, ["opcode_term_count"], ngram_min_df=2
    )
    assert first_audit == second_audit
    assert np.array_equal(first_creator, second_creator)
    for condition in ("with_creator", "without_creator"):
        for left, right in zip(first[condition]["opcode_term_count"], second[condition]["opcode_term_count"], strict=True):
            assert np.array_equal(left.toarray(), right.toarray())


def test_generated_with_creator_system_reproduces_phase7a() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    output = json.loads((config["output_dir"] / "phase7c_results.json").read_text(encoding="utf-8"))
    phase7a = json.loads(config["phase7a_results"].read_text(encoding="utf-8"))
    for period in ("P2", "P3", "P4", "P5"):
        current = next(
            item for item in output["records"]
            if item["condition"] == "with_creator"
            and item["prediction_period"] == period
        )
        expected = next(
            item["metrics"] for item in phase7a["scenarios"]
            if item["prediction_period"] == period
        )
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"):
            assert np.isclose(current[metric], expected[metric], atol=1e-12, rtol=0)
        assert [current[key] for key in ("tn", "fp", "fn", "tp")] == [
            expected[key] for key in ("tn", "fp", "fn", "tp")
        ]


def test_generated_feature_audits_have_no_placeholder_column() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    output = json.loads((config["output_dir"] / "phase7c_results.json").read_text(encoding="utf-8"))
    for audit in output["feature_audits"]:
        assert audit["creator_column_removed_not_replaced"] is True
        assert audit["static_columns_shared"] is True
        for dimensions in audit["dimensions"].values():
            assert dimensions["with_creator"] == dimensions["without_creator"] + 1
            assert dimensions["difference"] == 1
