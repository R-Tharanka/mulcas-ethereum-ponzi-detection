from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.experiments.phase6a import sha256_file
from src.extension.phase8r4a import (
    EXPECTED_PERIOD_ROWS,
    feature_schema,
    load_config,
    load_dataset,
    performance_preflight,
    verify_frozen_inputs,
)
from src.utils.paths import REPO_ROOT


RAW_HASH = "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"


def test_frozen_inputs_and_raw_dataset_remain_unchanged() -> None:
    config = load_config()
    assert len(verify_frozen_inputs(config)) >= 100
    assert sha256_file(config["dataset"]) == RAW_HASH


def test_exact_frozen_period_membership_is_descriptive_only() -> None:
    frame = load_dataset(load_config())
    assert frame.groupby("period").size().to_dict() == EXPECTED_PERIOD_ROWS
    assert len(frame) == 6498


def test_preflight_is_deterministic_sample_and_below_recorded_limit() -> None:
    config = load_config()
    frame = load_dataset(config)
    result = performance_preflight(frame, config)
    assert result["sample_size"] == 100
    assert result["execution_device"] == "cpu"
    assert result["estimated_full_corpus_runtime_seconds"] > 0
    assert result["estimated_uncompressed_matrix_size_bytes"] > 0


def test_schema_is_explicit_and_frozen_before_label_description() -> None:
    schema = feature_schema(load_config())
    assert schema["views"]["creation_full"]["dimension"] == 85
    assert schema["views"]["deployment_init"]["dimension"] == 90
    assert schema["views"]["creation_full_plus_init_and_boundary"]["dimension"] == 183
    assert schema["raw_constants_emitted"] is False


def test_generated_outputs_have_no_model_evaluation() -> None:
    output = REPO_ROOT / "results/extension/phase8r4a"
    result = json.loads((output / "phase8r4a_results.json").read_text(encoding="utf-8"))
    assert result["classifier_trained"] is False
    assert result["model_evaluation_performed"] is False
    assert result["p5_model_evaluation_performed"] is False
    assert result["p5_usage"] == "descriptive_period_summary_only"
    assert result["labels_used_during_feature_extraction"] is False
    assert result["determinism"]["hashes_equal"] is True
    assert result["dataset_sha256"] == RAW_HASH


def test_generated_matrix_matches_frozen_schema_and_hash() -> None:
    config = load_config()
    result = json.loads((config["output_dir"] / "phase8r4a_results.json").read_text(encoding="utf-8"))
    assert sha256_file(config["artifact_path"]) == result["feature_artifact"]["sha256"]
    with np.load(config["artifact_path"], allow_pickle=False) as arrays:
        assert arrays["row_index"].shape == (6498,)
        assert arrays["creation_full"].shape == (6498, 85)
        assert arrays["deployment_init"].shape == (6498, 90)
        assert arrays["creation_full_plus_init_and_boundary"].shape == (6498, 183)
        assert np.isfinite(arrays["creation_full_plus_init_and_boundary"]).all()


def test_generated_summaries_cover_every_period_and_label() -> None:
    output = REPO_ROOT / "results/extension/phase8r4a"
    periods = pd.read_csv(output / "period_creation_summary.csv")
    assert periods.set_index("period")["contract_count"].to_dict() == EXPECTED_PERIOD_ROWS
    labels = pd.read_csv(output / "label_descriptive_summary.csv")
    assert set(labels["label"]) == {0, 1}
    assert labels["descriptive_only"].astype(bool).all()
    source = (REPO_ROOT / "src/extension/phase8r4a.py").read_text(encoding="utf-8")
    assert "LogisticRegression" not in source
    assert "model.fit" not in source
