from __future__ import annotations

import json

import pandas as pd

from src.extension.phase8r5a import (
    EXPECTED_PERIOD_ROWS,
    load_config,
    load_sequence_corpus,
    runtime_estimates,
    temporal_vocabulary_audit,
    truncation_coverage,
    verify_frozen_inputs,
)
from src.utils.paths import REPO_ROOT


def test_all_frozen_anchors_and_raw_dataset_verify() -> None:
    config = load_config()
    observed = verify_frozen_inputs(config)
    assert len(observed) >= 125
    assert config["inputs"]["dataset"].exists()


def test_label_free_corpus_reuses_exact_periods() -> None:
    frame = load_sequence_corpus(load_config())
    assert len(frame) == 6498
    assert frame.groupby("period").size().to_dict() == EXPECTED_PERIOD_ROWS
    assert "label" not in frame.columns
    assert frame["tokens"].map(lambda values: isinstance(values, tuple)).all()


def test_temporal_vocabulary_never_fits_evaluation_tokens() -> None:
    config = load_config()
    frame = pd.DataFrame({
        "period": ["P0", "P1", "P2", "P3", "P4"],
        "tokens": [("OLD",), ("OLD",), ("P2_ONLY",), ("P3_ONLY",), ("P4_ONLY",)],
        "opcode_token_count": [1, 1, 1, 1, 1],
    })
    audit = temporal_vocabulary_audit(frame, config)
    p2 = audit[audit["evaluation_period"].eq("P2")].iloc[0]
    p3 = audit[audit["evaluation_period"].eq("P3")].iloc[0]
    assert p2["evaluation_oov_token_count"] == 1
    assert p3["evaluation_oov_token_count"] == 1
    assert not audit["evaluation_tokens_used_for_fit"].any()


def test_truncation_coverage_is_deterministic_and_label_independent() -> None:
    frame = pd.DataFrame({
        "period": ["P0", "P1", "P2", "P3", "P4", "P5"],
        "opcode_token_count": [0, 5, 10, 15, 20, 25],
    })
    first = truncation_coverage(frame, [10])
    second = truncation_coverage(frame, [10])
    pd.testing.assert_frame_equal(first, second)
    all_row = first[first["scope"].eq("ALL")].iloc[0]
    assert all_row["contracts_exceeding"] == 3
    assert all_row["retained_tokens_prefix_or_prefix_suffix"] == 45
    assert all_row["fixed_chunking_retained_fraction"] == 1.0


def test_runtime_estimate_uses_measured_1024_benchmark() -> None:
    config = load_config()
    frame = pd.DataFrame({
        "period": ["P0", "P1", "P2", "P3"],
        "opcode_token_count": [1024, 2048, 1024, 1024],
    })
    estimates = runtime_estimates(frame, [{"sequence_length": 1024, "examples_per_second": 2.0}], config)
    assert estimates["folds"][0]["estimated_chunks_per_epoch"] == 3
    assert estimates["folds"][0]["estimated_seconds_per_epoch_cpu"] == 1.5
    assert estimates["gpu_estimate"] is None


def test_generated_results_preserve_feasibility_guards() -> None:
    output = REPO_ROOT / "results/extension/phase8r5a"
    results = json.loads((output / "phase8r5a_results.json").read_text(encoding="utf-8"))
    assert results["contracts_audited"] == 6498
    assert results["p5_usage"] == "descriptive_only"
    assert results["p5_model_evaluation"] is False
    assert results["p5_in_model_design"] is False
    assert results["supervised_classifier_trained"] is False
    assert results["full_self_supervised_pretraining_run"] is False
    assert results["hypothesis_evaluated"] is False
    vocabulary = pd.read_csv(output / "temporal_vocabulary_audit.csv")
    assert vocabulary["evaluation_period"].tolist() == ["P2", "P3", "P4"]
    assert not vocabulary["evaluation_tokens_used_for_fit"].any()


def test_source_contains_no_supervised_classifier_or_full_training_loop() -> None:
    source = (REPO_ROOT / "src/extension/phase8r5a.py").read_text(encoding="utf-8")
    assert "LogisticRegression" not in source
    assert "XGBClassifier" not in source
    assert "for epoch" not in source
    loader = source[source.index("def load_sequence_corpus"):source.index("def _distribution")]
    assert '"label"' not in loader and "['label']" not in loader
