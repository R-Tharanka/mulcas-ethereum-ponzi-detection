"""Protocol and leakage tests for Phase 7A temporal robustness analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from src.evaluation.temporal import (
    build_ponzi_occurrence_periods,
    incremental_scenarios,
    validate_scenario_chronology,
)
from src.experiments.phase6a import verify_frozen_manifest
from src.experiments.phase7a import (
    DEFAULT_CONFIG_PATH,
    _load_frozen_configs,
    build_historical_feature_matrices,
    load_config,
    load_full_dataset,
)


def _assigned_dataset() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    config = load_config(DEFAULT_CONFIG_PATH)
    frame = load_full_dataset(config["dataset"])
    return build_ponzi_occurrence_periods(
        frame,
        positive_ends=config["periods"]["positive_ends"],
        expected_positives=config["periods"]["expected_total_positives"],
    )


def test_periods_are_chronological_disjoint_complete_and_keep_blocks() -> None:
    assigned, definitions = _assigned_dataset()
    assert len(assigned) == 6498
    assert assigned["row_index"].nunique() == 6498
    assert set(assigned["period"].value_counts().index) == {
        "P0", "P1", "P2", "P3", "P4", "P5"
    }
    assert assigned.groupby("createdBlockNumber")["period"].nunique().max() == 1
    for earlier, later in zip(definitions, definitions[1:], strict=False):
        assert int(earlier["block_max"]) < int(later["block_min"])


def test_period_positive_counts_and_scenarios_match_protocol() -> None:
    assigned, definitions = _assigned_dataset()
    assert [item["positive_count"] for item in definitions] == [50, 50, 50, 50, 50, 64]
    assert sum(int(item["row_count"]) for item in definitions) == len(assigned)
    scenarios = incremental_scenarios()
    assert [item.prediction_period for item in scenarios] == ["P2", "P3", "P4", "P5"]
    for scenario in scenarios:
        validate_scenario_chronology(assigned, scenario)


def test_feature_fitting_is_historical_only_sparse_and_label_safe() -> None:
    train = pd.DataFrame(
        {
            "opcode": ["PUSH1 ADD STOP", "PUSH1 SSTORE STOP", "CALL STOP", "ADD STOP"],
            "creator": ["a", "b", "c", "a"],
            "createdBlockNumber": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
        }
    )
    later = pd.DataFrame(
        {
            "opcode": ["FUTURE_ONLY STOP", "CALL STOP"],
            "creator": ["future", "b"],
            "createdBlockNumber": [5, 6],
            "label": [1, 0],
        }
    )
    views = ["opcode_term_count", "opcode_tfidf"]
    first, audit, creator = build_historical_feature_matrices(
        train, later, views, ngram_min_df=2
    )
    changed = later.copy()
    changed["label"] = 1 - changed["label"]
    second, changed_audit, changed_creator = build_historical_feature_matrices(
        train, changed, views, ngram_min_df=2
    )
    assert audit == changed_audit
    assert audit["historical_only"] is True
    assert audit["word2vec_fit"] is False
    assert np.array_equal(creator, changed_creator)
    for view in views:
        train_x, later_x = first[view]
        assert issparse(train_x) and issparse(later_x)
        assert np.array_equal(later_x.toarray(), second[view][1].toarray())


def test_frozen_mulcas_and_phase6_controls_remain_disabled() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    feature, model, mulcas = _load_frozen_configs(config)
    assert config["protocol"]["phase6_interventions_enabled"] is False
    assert config["protocol"]["optimization"] == "prohibited"
    assert mulcas["complexity_control_n"] == 2
    assert mulcas["voting_threshold"] == 0
    assert [item["name"] for item in mulcas["viewer_sequence"]] == [
        "xgboost_term_count",
        "xgboost_ngram_count_df_filtered",
        "xgboost_ngram_tfidf_df_filtered",
        "ridge_tfidf",
    ]
    assert model["models"]["ridge"]["class_weight"] is None
    assert model["models"]["svm"]["class_weight"] is None
    assert model["models"]["xgboost"].get("scale_pos_weight", 1) == 1
    assert feature["ngram"]["min_df"] == 2
    assert len(verify_frozen_manifest(config["frozen_manifest"])) == 13


def test_period_and_feature_outputs_are_deterministic() -> None:
    first, first_definitions = _assigned_dataset()
    second, second_definitions = _assigned_dataset()
    pd.testing.assert_frame_equal(first, second)
    assert first_definitions == second_definitions


def test_configured_scenarios_are_exact_and_no_later_period_trains_earlier() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    expected = incremental_scenarios()
    configured = config["scenarios"]
    assert len(configured) == len(expected)
    for item, scenario in zip(configured, expected, strict=True):
        assert tuple(item["train_periods"]) == scenario.train_periods
        assert item["prediction_period"] == scenario.prediction_period
        target_number = int(scenario.prediction_period[1:])
        assert all(int(period[1:]) < target_number for period in scenario.train_periods)
