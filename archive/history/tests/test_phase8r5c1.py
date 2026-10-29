from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.phase6a import sha256_file
from src.extension.phase8r5c1 import (
    FOLDS,
    SEEDS,
    _fit_probe,
    load_config,
    load_temporal_data,
    mean_seed_probabilities,
    verify_frozen_inputs,
)
from src.utils.paths import REPO_ROOT


def test_all_frozen_phase8r5b_artifacts_and_checkpoints_verify() -> None:
    frozen, checkpoints = verify_frozen_inputs(load_config())
    assert len(frozen) >= 166
    assert len(checkpoints) == 9
    for item in checkpoints:
        assert sha256_file(Path(item["path"])) == item["sha256"]


def test_configuration_freezes_scope_probe_and_ensemble() -> None:
    config = load_config()
    assert config["encoder_seeds"] == list(SEEDS)
    assert {key: tuple(value["historical_periods"]) for key, value in config["folds"].items()} == FOLDS
    assert config["embedding"]["dimension"] == 128
    assert config["embedding"]["sequence_length"] == 1024
    assert config["probe"]["C"] == 1.0
    assert config["probe"]["class_weight"] is None
    assert config["probe"]["threshold"] == 0.5
    assert config["ensemble"]["method"] == "arithmetic_mean_probability"
    assert config["execution"]["p5_allowed"] is False
    assert config["execution"]["transformer_fine_tuning"] is False


def test_temporal_loader_excludes_p5_and_reuses_exact_memberships() -> None:
    frame = load_temporal_data(load_config())
    assert len(frame) == 5989
    assert frame["row_index"].max() == 5988
    assert "P5" not in set(frame["period"])
    assert frame.groupby("period").size().to_dict() == {
        "P0": 104, "P1": 305, "P2": 5141, "P3": 289, "P4": 150,
    }


def test_three_seed_ensemble_is_exact_arithmetic_mean() -> None:
    values = [np.asarray([0.1, 0.9]), np.asarray([0.2, 0.8]), np.asarray([0.6, 0.4])]
    np.testing.assert_allclose(mean_seed_probabilities(values), np.asarray([0.3, 0.7]))


def test_fixed_probe_uses_native_half_threshold() -> None:
    config = load_config()
    train_x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    train_y = np.asarray([0, 0, 1, 1])
    score, prediction, _, _ = _fit_probe(train_x, train_y, np.asarray([[1.5], [10.0]]), config)
    np.testing.assert_array_equal(prediction, (score >= 0.5).astype(int))


def test_generated_outputs_are_complete_and_p5_free() -> None:
    output = REPO_ROOT / "results/extension/phase8r5c1"
    results = json.loads((output / "phase8r5c1_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "embedding_manifest.json").read_text(encoding="utf-8"))["embeddings"]
    contextual = pd.read_csv(output / "contextual_seed_metrics.csv")
    noncontextual = pd.read_csv(output / "noncontextual_seed_metrics.csv")
    ensembles = pd.read_csv(output / "contextual_ensemble_metrics.csv")
    assert results["scope"]["p5_loaded"] is False
    assert results["scope"]["transformer_fine_tuned"] is False
    if results["status"] == "stopped_preflight":
        assert results["h8r5_evaluated"] is False
        assert results["scope"]["classifier_trained"] is False
        assert results["preflight"]["full_execution_authorized"] is False
        assert not manifest and contextual.empty and noncontextual.empty and ensembles.empty
        return
    assert results["status"] == "complete"
    assert len(manifest) == 18
    assert len(contextual) == 9 and len(noncontextual) == 9 and len(ensembles) == 6
    assert set(contextual["seed"]) == set(SEEDS)
    assert set(contextual["period"]) == set(FOLDS)
    for item in manifest:
        path = Path(item["path"])
        assert path.is_file() and sha256_file(path) == item["sha256"]
        assert item["embedding_dimension"] == 128
        assert item["evaluation_period"] != "P5"
