from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.extension.phase8r3b import (
    EXPECTED_PERIOD_ROWS,
    _fit_predict,
    error_transitions,
    feature_schemas,
    load_config,
    load_frozen_predictions,
    select_candidate,
    subgroup_metrics,
    verify_frozen_inputs,
)
from src.utils.paths import REPO_ROOT


def test_frozen_inputs_verify_and_p5_is_never_loaded() -> None:
    config = load_config()
    assert len(verify_frozen_inputs(config)) >= 87
    frame = load_frozen_predictions(config)
    assert frame.groupby("period").size().to_dict() == EXPECTED_PERIOD_ROWS
    assert "P5" not in set(frame["period"])


def test_five_seed_mean_and_deployment_context_are_exact() -> None:
    config = load_config()
    frame = load_frozen_predictions(config)
    seed_columns = [f"gin_probability_seed_{seed}" for seed in config["inputs"]["gin_seeds"]]
    np.testing.assert_allclose(
        frame["gin_mean_probability"], frame[seed_columns].mean(axis=1), rtol=0, atol=1e-15
    )
    assert frame[["route_first_viewer", "route_second_viewer", "route_voting_stage"]].sum(axis=1).eq(1).all()
    assert frame["historically_novel_code"].eq(~frame["has_any_exact_historical_clone"].astype(bool)).all()
    assert frame["unseen_creator"].eq(~frame["previously_seen_creator"].astype(bool)).all()


def test_only_predeclared_feature_sets_and_no_period_or_identity() -> None:
    schemas = feature_schemas(load_config())
    assert list(schemas) == ["hybrid_a", "hybrid_b"]
    prohibited = {"period", "row_index", "address", "label", "error_type"}
    assert not prohibited.intersection(schemas["hybrid_a"])
    assert not prohibited.intersection(schemas["hybrid_b"])
    assert len(schemas["hybrid_a"]) == 3
    assert len(schemas["hybrid_b"]) == 9


def test_forward_protocol_scaler_and_selection_are_temporal_only() -> None:
    config = load_config()
    frame = load_frozen_predictions(config)
    schemas = feature_schemas(config)
    p2 = frame[frame["period"].eq("P2")]
    p3 = frame[frame["period"].eq("P3")]
    _, _, diagnostic = _fit_predict(p2, p3, schemas["hybrid_a"], config)
    assert diagnostic["scaler_fit_row_indices"] == p2["row_index"].astype(int).tolist()
    assert set(diagnostic["scaler_fit_row_indices"]).isdisjoint(set(p3["row_index"]))
    candidates = [
        {"system": "hybrid_a", "precision": 0.81, "f1": 0.4, "recall": 0.3, "balanced_accuracy": 0.6},
        {"system": "hybrid_b", "precision": 0.82, "f1": 0.4, "recall": 0.3, "balanced_accuracy": 0.6},
    ]
    selected = select_candidate(candidates, config)
    assert selected["feature_set"] == "hybrid_a"
    assert selected["selection_period"] == "P3"


def test_meta_model_is_fixed_unweighted_and_native_threshold() -> None:
    config = load_config()
    model = config["meta_classifier"]
    assert model == {
        "family": "LogisticRegression", "penalty": "l2", "C": 1.0,
        "class_weight": None, "solver": "liblinear", "max_iter": 2000,
        "threshold": 0.5, "scaler": "StandardScaler",
    }
    source = (REPO_ROOT / "src/extension/phase8r3b.py").read_text(encoding="utf-8")
    assert "MulCasCascade" not in source
    assert "GraphDataLoader" not in source
    assert "threshold_optimization" not in source
    assert "train_test_split" not in source


def test_generated_outputs_reconcile_and_record_guards() -> None:
    output = REPO_ROOT / "results/extension/phase8r3b"
    result = json.loads((output / "phase8r3b_results.json").read_text(encoding="utf-8"))
    assert result["base_models_retrained"] is False
    assert result["threshold_tuned"] is False
    assert result["p5_loaded_or_evaluated"] is False
    assert result["selection"]["selection_period"] == "P3"
    confirmation = pd.read_csv(output / "p4_confirmation_metrics.csv")
    assert "delta_f1_vs_mulcas" in confirmation
    assert "delta_f1_vs_gin" in confirmation
    predictions = pd.read_csv(output / "hybrid_predictions.csv")
    assert set(predictions["period"]) == {"P3", "P4"}
    assert len(predictions[predictions["period"].eq("P3")]) == 2 * EXPECTED_PERIOD_ROWS["P3"]
    assert len(predictions[predictions["period"].eq("P4")]) == EXPECTED_PERIOD_ROWS["P4"]
    transitions = error_transitions(predictions)
    subgroups = subgroup_metrics(predictions)
    pd.testing.assert_frame_equal(transitions, pd.read_csv(output / "error_transition.csv"))
    pd.testing.assert_frame_equal(subgroups, pd.read_csv(output / "subgroup_metrics.csv"))


def test_generated_predictions_are_deterministic_for_fixed_seed() -> None:
    config = load_config()
    frame = load_frozen_predictions(config)
    p2 = frame[frame["period"].eq("P2")]
    p3 = frame[frame["period"].eq("P3")]
    columns = feature_schemas(config)["hybrid_b"]
    first_score, first_prediction, _ = _fit_predict(p2, p3, columns, config)
    second_score, second_prediction, _ = _fit_predict(p2, p3, columns, config)
    np.testing.assert_allclose(first_score, second_score, rtol=0, atol=0)
    np.testing.assert_array_equal(first_prediction, second_prediction)
