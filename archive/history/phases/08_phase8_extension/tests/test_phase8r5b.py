from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import torch

from src.extension.phase8r5b import (
    EXPECTED_FOLDS,
    EXPECTED_SEEDS,
    load_config,
    verify_cuda,
    verify_frozen_inputs,
)
from src.extension.sequence_pretraining import load_p0_p4_sequences, prepare_fold_corpus
from src.experiments.phase6a import sha256_file
from src.utils.paths import REPO_ROOT


def test_all_frozen_phase8r5b0_anchors_verify() -> None:
    observed = verify_frozen_inputs(load_config())
    assert len(observed) >= 143


def test_configuration_freezes_approved_scope() -> None:
    config = load_config()
    assert config["random_seeds"] == EXPECTED_SEEDS
    assert {key: tuple(value["historical_periods"]) for key, value in config["folds"].items()} == EXPECTED_FOLDS
    assert config["sequence"]["length"] == 1024
    assert config["masking"]["probability"] == 0.15
    assert config["optimization"]["epochs"] == 10
    assert config["optimization"]["early_stopping"] is False
    assert config["execution"]["amp"] is True
    assert config["execution"]["physical_batch_size"] == 8
    assert config["execution"]["cpu_fallback"] is False
    assert config["execution"]["p5_allowed"] is False
    assert config["execution"]["labels_allowed"] is False
    assert config["execution"]["supervised_training"] is False


def test_loader_excludes_p5_and_labels_and_reuses_periods() -> None:
    config = load_config()
    frame = load_p0_p4_sequences(config["inputs"]["dataset"], config["inputs"]["period_definitions"])
    assert len(frame) == 5989
    assert frame["row_index"].max() == 5988
    assert "label" not in frame.columns
    assert "P5" not in set(frame["period"])
    assert frame.groupby("period").size().to_dict() == {
        "P0": 104, "P1": 305, "P2": 5141, "P3": 289, "P4": 150,
    }


def test_fold_corpora_exclude_the_future_period() -> None:
    config = load_config()
    frame = load_p0_p4_sequences(config["inputs"]["dataset"], config["inputs"]["period_definitions"])
    for fold, historical in EXPECTED_FOLDS.items():
        corpus = prepare_fold_corpus(frame, fold=fold, historical_periods=historical, chunk_length=1024)
        future_rows = set(frame.loc[frame["period"].eq(fold), "row_index"])
        assert future_rows.isdisjoint(corpus.row_indices)
        assert set(frame.loc[frame["row_index"].isin(corpus.row_indices), "period"]) == set(historical)


def test_cuda_gate_matches_approved_environment() -> None:
    environment = verify_cuda(load_config())
    assert environment["cuda_available"] is True
    assert environment["device_name"] == "NVIDIA GeForce RTX 2050"
    assert environment["amp"] is True


def test_source_has_no_label_or_p5_training_and_no_classifier() -> None:
    root = Path(__file__).resolve().parents[1]
    pretraining = (root / "src/extension/sequence_pretraining.py").read_text(encoding="utf-8")
    runner = (root / "src/extension/phase8r5b.py").read_text(encoding="utf-8")
    loader = pretraining[pretraining.index("def load_p0_p4_sequences"):pretraining.index("def prepare_fold_corpus")]
    assert '"label"' not in loader
    assert "nrows=P0_P4_ROWS" in loader
    assert "classification" not in pretraining.lower()
    assert '"P5"' not in EXPECTED_FOLDS
    assert "early_stopping" in runner


def test_generated_outputs_contain_all_nine_complete_runs() -> None:
    output = REPO_ROOT / "results/extension/phase8r5b"
    results = json.loads((output / "phase8r5b_results.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(output / "pretraining_runs.csv")
    epochs = pd.read_csv(output / "epoch_metrics.csv")
    assert results["status"] == "complete"
    assert results["runs_completed"] == 9
    assert results["p5_loaded"] is False
    assert results["ponzi_labels_loaded"] is False
    assert results["supervised_classifier_trained"] is False
    assert results["h8r5_evaluated"] is False
    assert set(runs["fold"]) == {"P2", "P3", "P4"}
    assert set(runs["seed"]) == set(EXPECTED_SEEDS)
    assert runs.groupby("fold").size().to_dict() == {"P2": 3, "P3": 3, "P4": 3}
    assert len(epochs) == 90
    assert epochs.groupby(["fold", "seed"]).size().eq(10).all()
    assert epochs["mlm_training_loss"].map(lambda value: value == value).all()


def test_checkpoint_hashes_metadata_and_independent_initializations() -> None:
    output = REPO_ROOT / "results/extension/phase8r5b"
    manifest = json.loads((output / "checkpoint_manifest.json").read_text(encoding="utf-8"))["checkpoints"]
    runs = pd.read_csv(output / "pretraining_runs.csv")
    assert len(manifest) == 9
    assert runs.groupby("fold")["initial_model_state_hash"].nunique().eq(3).all()
    assert runs["fresh_initialization"].all()
    assert not runs["optimizer_state_reused"].any()
    period_limits = {"P2": 408, "P3": 5549, "P4": 5838}
    for item in manifest:
        path = Path(item["path"])
        assert path.is_file()
        assert sha256_file(path) == item["sha256"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        assert metadata["fold"] == item["fold"]
        assert metadata["seed"] == item["seed"]
        assert metadata["epoch"] == 10
        assert max(metadata["historical_row_indices"]) == period_limits[item["fold"]]
        assert metadata["architecture"]["classification_head"] is False
        assert metadata["sequence_policy"]["length"] == 1024
        assert metadata["masking_policy"]["probability"] == 0.15
        assert metadata["amp"] is True
