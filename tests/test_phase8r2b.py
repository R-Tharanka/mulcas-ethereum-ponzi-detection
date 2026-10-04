from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.extension.phase8r2b import (
    architecture_grid,
    load_config,
    load_temporal_frame,
    verify_frozen_inputs,
)


def test_frozen_phase8r2a_chain_and_internal_hashes_verify() -> None:
    config = load_config()
    frozen = verify_frozen_inputs(config)
    assert frozen
    manifest = json.loads(config["graph_manifest"].read_text(encoding="utf-8"))
    assert manifest["corpus_sha256"] == config["frozen_graph"]["corpus_sha256"]
    assert manifest["schema_sha256"] == config["frozen_graph"]["schema_sha256"]


def test_only_exact_p0_to_p4_memberships_are_loaded() -> None:
    frame = load_temporal_frame(load_config())
    assert len(frame) == 5989
    assert frame["row_index"].to_numpy().max() == 5988
    assert set(frame["period"]) == {"P0", "P1", "P2", "P3", "P4"}
    assert "P5" not in set(frame["period"])


def test_grid_seeds_threshold_and_imbalance_policy_are_frozen() -> None:
    config = load_config()
    grid = architecture_grid(config)
    assert len(grid) == 8
    assert {(x["layers"], x["hidden_dim"], x["dropout"]) for x in grid} == {
        (layers, hidden, dropout) for layers in (2, 3) for hidden in (64, 128) for dropout in (0.2, 0.5)
    }
    assert config["gin"]["seeds"] == [7, 42, 123, 2026, 999]
    assert config["gin"]["threshold"] == 0.5
    assert config["gin"]["loss"] == "BCEWithLogitsLoss"
    assert config["gin"]["pos_weight"] == "inner_train_negative_to_positive_ratio"
    assert config["aggregate_baseline"]["class_weight"] is None
    assert not any(key in config for key in ("sampling", "smote", "threshold_search"))


def test_pos_weight_formula_uses_inner_training_labels_only() -> None:
    inner = np.asarray([0, 0, 0, 1])
    future = np.asarray([1, 1, 1, 1, 1])
    weight = float((inner == 0).sum() / (inner == 1).sum())
    assert weight == 3.0
    assert weight != float((np.r_[inner, future] == 0).sum() / (np.r_[inner, future] == 1).sum())


def test_generated_outputs_are_complete_and_exclude_p5() -> None:
    config = load_config()
    output = config["output_dir"]
    runs = pd.read_csv(output / "gin_runs.csv")
    predictions = pd.read_csv(output / "gin_predictions.csv")
    diagnostics = pd.read_csv(output / "training_diagnostics.csv")
    assert len(runs) == 8 * 5 * 3
    assert len(predictions) == 40 * (5141 + 289 + 150)
    assert len(diagnostics) == len(runs)
    assert set(runs["seed"]) == {7, 42, 123, 2026, 999}
    assert set(predictions["period"]) == {"P2", "P3", "P4"}
    assert (runs["threshold"] == 0.5).all()
    assert (~diagnostics["future_used_for_early_stopping"].astype(bool)).all()
    prediction_minimums = predictions.groupby("period")["row_index"].min().to_dict()
    for row in diagnostics.itertuples(index=False):
        assert int(row.normalizer_fit_row_max) < int(prediction_minimums[row.prediction_period])
