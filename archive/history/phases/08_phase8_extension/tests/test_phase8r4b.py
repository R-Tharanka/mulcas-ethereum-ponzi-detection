from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.experiments.phase6a import sha256_file
from src.extension.phase8r4b import (
    EXPECTED_ROWS,
    fit_predict,
    load_config,
    load_diagnostics,
    load_frozen_views,
    load_p0_p4_metadata,
    select_candidate,
    verify_frozen_inputs,
)
from src.utils.paths import REPO_ROOT


MATRIX_HASH = "ecc80b2ab83c5954eb8790e7f693619441116506d64f8c6e3aa8aec87bcdf897"


def test_frozen_chain_and_creation_matrix_hash_verify() -> None:
    config = load_config()
    assert len(verify_frozen_inputs(config)) >= 117
    assert sha256_file(config["inputs"]["creation_feature_matrix"]) == MATRIX_HASH


def test_exact_p0_p4_membership_is_loaded_without_p5() -> None:
    metadata = load_p0_p4_metadata(load_config())
    assert len(metadata) == 5989
    assert metadata.groupby("period").size().to_dict() == EXPECTED_ROWS
    assert "P5" not in set(metadata["period"])


def test_frozen_views_are_exact_and_unavailable_init_row_remains_zero() -> None:
    config = load_config()
    metadata = load_p0_p4_metadata(config)
    views = load_frozen_views(metadata, config)
    assert {name: matrix.shape for name, matrix in views.items()} == {
        "creation_full": (5989, 85), "deployment_init": (5989, 90),
        "combined": (5989, 183),
    }
    position = metadata.index[metadata["row_index"].eq(986)].item()
    assert np.count_nonzero(views["deployment_init"][position]) == 0
    assert np.isfinite(views["combined"]).all()


def test_model_configuration_is_fixed_unweighted_and_native_threshold() -> None:
    config = load_config()
    logistic = config["models"]["logistic_regression"]
    xgboost = config["models"]["xgboost"]
    assert logistic["C"] == 1.0 and logistic["class_weight"] is None
    assert logistic["penalty"] == "l2" and logistic["threshold"] == 0.5
    assert xgboost["scale_pos_weight"] == 1.0 and xgboost["threshold"] == 0.5
    assert xgboost["n_estimators"] == 100 and xgboost["max_depth"] == 3
    assert xgboost["learning_rate"] == 0.1 and xgboost["n_jobs"] == 1


def test_scaler_fits_historical_rows_only_and_results_are_deterministic() -> None:
    config = load_config()
    metadata = load_p0_p4_metadata(config)
    matrix = load_frozen_views(metadata, config)["creation_full"]
    train_mask = metadata["period"].isin(["P0", "P1"]).to_numpy()
    target_mask = metadata["period"].eq("P2").to_numpy()
    args = (
        "logistic_regression", matrix[train_mask],
        metadata.loc[train_mask, "label"].to_numpy(dtype=int), matrix[target_mask],
        metadata.loc[train_mask, "row_index"].to_numpy(dtype=int), config,
    )
    first_score, first_prediction, first_detail = fit_predict(*args)
    second_score, second_prediction, second_detail = fit_predict(*args)
    expected_rows = metadata.loc[train_mask, "row_index"].astype(int).tolist()
    assert first_detail["scaler_fit_row_indices"] == expected_rows
    assert set(expected_rows).isdisjoint(set(metadata.loc[target_mask, "row_index"]))
    np.testing.assert_allclose(first_score, second_score, rtol=0, atol=0)
    np.testing.assert_array_equal(first_prediction, second_prediction)
    assert first_detail["threshold"] == second_detail["threshold"] == 0.5


def test_diagnostic_join_is_exact_and_leakage_safe() -> None:
    config = load_config()
    metadata = load_p0_p4_metadata(config)
    diagnostics = load_diagnostics(metadata, config)
    assert len(diagnostics) == 5580
    assert diagnostics["row_index"].nunique() == 5580
    assert set(diagnostics["period"]) == {"P2", "P3", "P4"}
    assert "P5" not in set(diagnostics["period"])
    assert diagnostics["previously_seen_creator"].notna().all()


def test_selection_uses_macro_metrics_and_fixed_simplicity_tie_break() -> None:
    summary = pd.DataFrame([
        {"candidate_id": "logistic_regression__creation_full", "model": "logistic_regression", "feature_view": "creation_full", "eligible": True, "macro_mean_f1": 0.5, "macro_mean_recall": 0.4, "macro_mean_balanced_accuracy": 0.6, "simplicity_rank": 0},
        {"candidate_id": "xgboost__combined", "model": "xgboost", "feature_view": "combined", "eligible": True, "macro_mean_f1": 0.5, "macro_mean_recall": 0.4, "macro_mean_balanced_accuracy": 0.6, "simplicity_rank": 5},
    ])
    assert select_candidate(summary)["candidate_id"] == "logistic_regression__creation_full"


def test_generated_outputs_record_all_guards_and_reconcile() -> None:
    output = REPO_ROOT / "results/extension/phase8r4b"
    result = json.loads((output / "phase8r4b_results.json").read_text(encoding="utf-8"))
    assert result["p5_loaded_or_evaluated"] is False
    assert result["mulcas_retrained"] is False
    assert result["feature_selection_performed"] is False
    assert result["threshold_tuned"] is False
    assert result["class_weighting"] == "none" and result["sampling"] == "none"
    assert result["matrix_sha256"] == MATRIX_HASH
    metrics = pd.read_csv(output / "model_period_metrics.csv")
    summary = pd.read_csv(output / "candidate_summary.csv")
    predictions = pd.read_csv(output / "creation_predictions.csv")
    assert len(metrics) == 18 and len(summary) == 6
    assert set(metrics["period"]) == {"P2", "P3", "P4"}
    assert "P5" not in set(predictions["period"])
    assert len(predictions) == 6 * 5580
    assert predictions.groupby(["candidate_id", "period"]).size().groupby(level=1).first().to_dict() == {"P2": 5141, "P3": 289, "P4": 150}


def test_source_has_no_sampling_feature_selection_or_mulcas_training() -> None:
    source = (REPO_ROOT / "src/extension/phase8r4b.py").read_text(encoding="utf-8")
    assert "SelectKBest" not in source
    assert "RandomOverSampler" not in source and "RandomUnderSampler" not in source
    assert "MulCasCascade" not in source
    assert "train_test_split" not in source
