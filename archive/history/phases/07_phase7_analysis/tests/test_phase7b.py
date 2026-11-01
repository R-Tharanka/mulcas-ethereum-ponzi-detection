"""Protocol tests for Phase 7B frozen MulCas component ablation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from src.evaluation.temporal import build_ponzi_occurrence_periods
from src.experiments.phase6a import verify_frozen_manifest
from src.experiments.phase7a import (
    _load_frozen_configs,
    build_historical_feature_matrices,
    load_config as load_phase7a_config,
    load_full_dataset,
)
from src.experiments.phase7b import (
    DEFAULT_CONFIG_PATH,
    build_ablation_predictions,
    describe_frozen_viewers,
    load_config,
)
from src.models.mulcas import ViewerPrediction


def _output(index: list[int], scores: list[float], predictions: list[int]) -> ViewerPrediction:
    return ViewerPrediction(
        probability=pd.Series(scores, index=index, dtype=float),
        prediction=pd.Series(predictions, index=index, dtype=int),
    )


def _synthetic_outputs(mulcas: dict) -> tuple[pd.DataFrame, dict]:
    index = [10, 11, 12, 13]
    first = mulcas["viewer_sequence"][0]["name"]
    second = mulcas["viewer_sequence"][1]["name"]
    third = mulcas["viewer_sequence"][2]["name"]
    fourth = mulcas["viewer_sequence"][3]["name"]
    cascade = {
        first: _output(index, [0.01, 0.40, 0.60, 0.99], [0, 0, 1, 1]),
        second: _output(index, [0.02, 0.04, 0.50, 0.50], [0, 0, 1, 0]),
    }
    voting = {
        first: _output(index, [0.01, 0.40, 0.60, 0.99], [0, 0, 1, 1]),
        second: _output(index, [0.02, 0.10, 0.70, 0.40], [0, 0, 1, 0]),
        third: _output(index, [0.10, 0.20, 0.80, 0.30], [0, 0, 1, 0]),
        fourth: _output(index, [0.20, 0.30, 0.70, 0.40], [0, 0, 1, 0]),
    }
    full = pd.DataFrame(
        {
            "sample_index": index,
            "prediction": [0, 0, 1, 1],
            "probability": [0.01, 0.40, 0.80, 0.99],
            "handled_by": [first, second, "vote", first],
        }
    )
    return full, {"cascade": cascade, "voting": voting}


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


def test_actual_viewer_description_and_frozen_thresholds() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    _, models, mulcas = _load_frozen_configs(load_phase7a_config(config["phase7a_config"]))
    description = describe_frozen_viewers(mulcas)
    assert description["cascade_depth"] == 2
    assert description["voting_threshold"] == 0
    assert [(item["classifier_family"], item["cascade_stage"]) for item in description["viewers"]] == [
        ("xgboost", 1), ("xgboost", 2), ("xgboost", None), ("ridge", None)
    ]
    assert description["viewers"][0]["negative_threshold"] == 0.05
    assert description["viewers"][0]["positive_threshold"] == 0.95
    assert description["viewers"][1]["positive_threshold"] == 0.51
    assert models["models"]["ridge"]["class_weight"] is None
    assert models["models"]["xgboost"].get("scale_pos_weight", 1) == 1


def test_each_ablation_changes_only_its_declared_component() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    _, _, mulcas = _load_frozen_configs(load_phase7a_config(config["phase7a_config"]))
    full, outputs = _synthetic_outputs(mulcas)
    results = build_ablation_predictions(
        full["sample_index"].to_numpy(), mulcas, full, outputs
    )
    first = mulcas["viewer_sequence"][0]["name"]
    second = mulcas["viewer_sequence"][1]["name"]
    assert list(results) == [item["id"] for item in config["ablations"]]
    pd.testing.assert_frame_equal(results["full_frozen_mulcas"], full)
    assert first not in set(results["remove_first_cascade_viewer"]["handled_by"])
    assert second not in set(results["remove_second_cascade_viewer"]["handled_by"])
    assert "vote" not in set(results["remove_voting_stage"]["handled_by"])
    assert set(results["first_viewer_only"]["handled_by"]) == {f"standalone__{first}"}
    assert set(results["second_viewer_only"]["handled_by"]) == {f"standalone__{second}"}


def test_no_voting_fallback_and_ablation_outputs_are_deterministic() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    _, _, mulcas = _load_frozen_configs(load_phase7a_config(config["phase7a_config"]))
    full, outputs = _synthetic_outputs(mulcas)
    first = build_ablation_predictions(full["sample_index"].to_numpy(), mulcas, full, outputs)
    second = build_ablation_predictions(full["sample_index"].to_numpy(), mulcas, full, outputs)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    no_vote = first["remove_voting_stage"].set_index("sample_index")
    assert no_vote.loc[12, "handled_by"] == "fallback_last_cascade_viewer"
    second_viewer = mulcas["viewer_sequence"][1]["name"]
    assert no_vote.loc[12, "prediction"] == outputs["cascade"][second_viewer].prediction.loc[12]


def test_feature_fitting_and_creator_state_remain_historical_only() -> None:
    train = pd.DataFrame(
        {
            "opcode": ["PUSH1 ADD", "CALL STOP", "PUSH1 SSTORE", "ADD STOP"],
            "creator": ["a", "b", "a", "c"],
            "createdBlockNumber": [1, 2, 3, 4],
            "label": [0, 1, 1, 0],
        }
    )
    future = pd.DataFrame(
        {
            "opcode": ["FUTURE_ONLY STOP", "CALL STOP"],
            "creator": ["b", "future"],
            "createdBlockNumber": [5, 6],
            "label": [0, 1],
        }
    )
    views = ["opcode_term_count", "opcode_tfidf"]
    first, audit, creator = build_historical_feature_matrices(
        train, future, views, ngram_min_df=2
    )
    changed = future.assign(label=1 - future["label"])
    second, second_audit, second_creator = build_historical_feature_matrices(
        train, changed, views, ngram_min_df=2
    )
    assert audit == second_audit and audit["historical_only"]
    assert np.array_equal(creator, second_creator)
    for view in views:
        assert issparse(first[view][0]) and issparse(first[view][1])
        assert np.array_equal(first[view][1].toarray(), second[view][1].toarray())


def test_previous_artifacts_are_frozen_and_no_optimization_is_enabled() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert len(verify_frozen_manifest(config["frozen_manifest"])) == 11
    assert config["protocol"]["optimization"] == "prohibited"
    assert config["protocol"]["retuning"] == "prohibited"
    assert config["protocol"]["phase6_interventions_enabled"] is False
    assert config["ablation_semantics"] == {
        "removed_viewer_scope": "cascade_and_voting",
        "no_voting_fallback": "last_cascade_viewer_native_prediction",
        "standalone_viewer_source": "full_historical_training_voting_copy",
        "standalone_decision": "native_model_prediction",
    }


def test_generated_full_system_reproduces_phase7a_metrics() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    output = json.loads((config["output_dir"] / "phase7b_results.json").read_text(encoding="utf-8"))
    phase7a = json.loads(config["phase7a_results"].read_text(encoding="utf-8"))
    for period in ("P2", "P3", "P4", "P5"):
        current = next(
            item for item in output["records"]
            if item["ablation_id"] == "full_frozen_mulcas"
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
