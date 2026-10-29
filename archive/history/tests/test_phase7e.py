"""Protocol tests for Phase 7E historical clone sensitivity analysis."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from src.experiments.phase6a import sha256_file
from src.experiments.phase7e import (
    DEFAULT_CONFIG_PATH,
    NOVELTY_GROUPS,
    build_class_analysis,
    build_clone_membership,
    build_group_metrics,
    build_routing_interaction,
    load_config,
    normalize_bytecode,
    normalize_opcode_sequence,
    safe_group_metrics,
    validate_partition_counts,
    verify_prior_phase_chain,
)


def _small_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": range(8),
            "period": ["P0", "P1", "P2", "P2", "P3", "P4", "P5", "P5"],
            "createdBlockNumber": [1, 2, 3, 3, 4, 5, 6, 6],
            "opcode": ["add stop", "MUL", " ADD   STOP ", "FUTURE", "FUTURE", "MUL", "PUSH1", "PUSH1"],
            "contractCode": ["0x0100", "0x02", "0100", "0x99", "0x99", "02", "0x60aa", "0x60aa"],
            "creationCode": ["0xaaaa", "0xbb", "AAAA", "0xcc", "cc", "0xbb", "0xdd", "0xdd"],
            "creator": ["a", "b", "a", "c", "c", "b", "d", "d"],
            "label": [0, 1, 1, 0, 1, 0, 1, 0],
        }
    )


def test_phase7e_configuration_freezes_model_and_matching_scope() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    protocol = config["protocol"]
    assert protocol["optimization"] == "prohibited"
    assert protocol["retraining"] == "prohibited"
    assert protocol["retuning"] == "prohibited"
    assert protocol["fuzzy_matching"] == "prohibited"
    assert protocol["historical_scope"] == "strictly_earlier_training_periods"
    assert protocol["labels_used_for_clone_status"] is False
    mulcas = yaml.safe_load(config["paper_mulcas_config"].read_text(encoding="utf-8"))
    assert mulcas["complexity_control_n"] == 2
    assert mulcas["voting_threshold"] == 0
    assert [
        (viewer["negative_threshold"], viewer["positive_threshold"])
        for viewer in mulcas["viewer_sequence"][:2]
    ] == [(0.05, 0.95), (0.05, 0.51)]


def test_exact_normalization_is_deterministic_not_fuzzy() -> None:
    assert normalize_opcode_sequence(" add   PUSH1 stop ") == "ADD PUSH1 STOP"
    assert normalize_opcode_sequence("ADD PUSH1 STOP") == "ADD PUSH1 STOP"
    assert normalize_opcode_sequence("ADD PUSH2 STOP") != "ADD PUSH1 STOP"
    assert normalize_bytecode("0xAA01") == "aa01"
    assert normalize_bytecode("aa01") == "aa01"
    assert normalize_bytecode("aa02") != "aa01"
    assert normalize_bytecode("not-hex") == ""


def test_clone_matching_uses_only_strictly_earlier_periods() -> None:
    source = _small_history()
    membership = build_clone_membership(source)
    p2_clone = membership[membership["row_index"] == 2].iloc[0]
    p2_same_period = membership[membership["row_index"] == 3].iloc[0]
    p3 = membership[membership["row_index"] == 4].iloc[0]
    p5_first = membership[membership["row_index"] == 6].iloc[0]
    p5_second = membership[membership["row_index"] == 7].iloc[0]
    assert p2_clone["historical_opcode_match_count"] == 1
    assert p2_clone["earliest_historical_opcode_match_block"] == 1
    assert p2_clone["latest_historical_opcode_match_block"] == 1
    assert p2_clone["novelty_group"] == "exact_runtime_clone"
    assert p2_same_period["novelty_group"] == "no_exact_historical_clone"
    assert p3["historical_runtime_bytecode_match_count"] == 1
    assert p5_first["novelty_group"] == "no_exact_historical_clone"
    assert p5_second["novelty_group"] == "no_exact_historical_clone"


def test_clone_status_is_label_independent_and_future_independent() -> None:
    source = _small_history()
    first = build_clone_membership(source)
    changed = source.assign(label=1 - source["label"])
    changed.loc[changed["period"] == "P5", ["opcode", "contractCode", "creationCode"]] = ["ADD STOP", "0x0100", "0xaaaa"]
    second = build_clone_membership(changed)
    columns = [column for column in first if column != "previously_seen_creator"]
    pd.testing.assert_frame_equal(
        first[first["period"].isin(["P2", "P3", "P4"])][columns].reset_index(drop=True),
        second[second["period"].isin(["P2", "P3", "P4"])][columns].reset_index(drop=True),
    )


def test_undefined_subgroup_metrics_are_null_not_forced() -> None:
    frame = pd.DataFrame(
        {"label": [0, 0], "final_prediction": [0, 0], "final_score": [0.1, 0.2]}
    )
    metrics = safe_group_metrics(frame)
    assert metrics["recall"] is None
    assert metrics["precision"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["pr_auc"] is None


def test_every_generated_row_has_one_novelty_group() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    membership = pd.read_csv(config["output_dir"] / "phase7e_clone_membership.csv")
    predictions = pd.read_csv(config["phase7a_predictions"])
    assert len(membership) == len(predictions) == 6089
    assert membership["row_index"].is_unique
    assert set(membership["novelty_group"]).issubset(NOVELTY_GROUPS)
    assert membership["novelty_group"].notna().all()


def test_group_confusion_counts_sum_to_phase7a_period_totals() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    metrics = pd.read_csv(config["output_dir"] / "phase7e_group_metrics.csv")
    membership = pd.read_csv(config["output_dir"] / "phase7e_clone_membership.csv")
    validate_partition_counts(membership, metrics)
    phase7a = json.loads(config["phase7a_results"].read_text(encoding="utf-8"))
    for scenario in phase7a["scenarios"]:
        current = metrics[
            (metrics["period"] == scenario["prediction_period"])
            & (metrics["clone_group"] == "all_contracts")
        ].iloc[0]
        for key in ("tn", "fp", "fn", "tp"):
            assert int(current[key]) == int(scenario["metrics"][key])


def test_generated_summaries_are_deterministic() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    inventory = pd.read_csv(config["output_dir"] / "phase7e_clone_membership.csv")
    inventory["reached_voting_stage"] = inventory["reached_voting_stage"].astype(bool)
    inventory["has_any_exact_historical_clone"] = inventory["has_any_exact_historical_clone"].astype(bool)
    pd.testing.assert_frame_equal(build_group_metrics(inventory), build_group_metrics(inventory))
    pd.testing.assert_frame_equal(build_class_analysis(inventory), build_class_analysis(inventory))
    pd.testing.assert_frame_equal(build_routing_interaction(inventory), build_routing_interaction(inventory))


def test_prior_artifacts_and_phase7a_predictions_remain_frozen() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    frozen = verify_prior_phase_chain(config["frozen_manifest"])
    assert "results/experiments/phase7a/phase7a_predictions.csv" in frozen
    assert "results/experiments/phase7d/phase7d_results.json" in frozen
    assert "results/baseline/paper_aligned/mulcas_results.json" in frozen
    results = json.loads((config["output_dir"] / "phase7e_results.json").read_text(encoding="utf-8"))
    assert sha256_file(config["phase7a_predictions"]) == results["phase7a_predictions_sha256"]
    assert results["model_training_performed"] is False
    assert results["optimization_performed"] is False
    assert results["fuzzy_matching_performed"] is False
