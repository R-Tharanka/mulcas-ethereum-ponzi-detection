"""Protocol tests for Phase 7D frozen-prediction error analysis."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.experiments.phase7d import (
    DEFAULT_CONFIG_PATH,
    add_contract_diagnostics,
    add_historical_context,
    build_clone_summary,
    build_period_summary,
    build_routing_summary,
    classify_outcomes,
    load_config,
    parse_boolean_series,
    verify_prior_phase_chain,
)


def _context_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "period": ["P0", "P1", "P2", "P3"],
            "createdBlockNumber": [1, 2, 3, 4],
            "creator": ["a", "b", "b", "future"],
            "label": [1, 0, 1, 1],
            "opcode": ["ADD", "STOP", "ADD", "ADD"],
            "contractCode": ["0x01", "0x00", "0x01", "0x01"],
            "creationCode": ["0xaa", "0xbb", "0xaa", "0xaa"],
        }
    )
    inventory = assigned.loc[assigned["period"].isin(["P2", "P3"])].copy()
    inventory["creator_history_feature"] = [0, 0]
    return inventory, assigned


def test_phase7d_configuration_prohibits_model_changes() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config["protocol"]["optimization"] == "prohibited"
    assert config["protocol"]["retraining"] == "prohibited"
    assert config["protocol"]["retuning"] == "prohibited"
    assert config["protocol"]["reference_predictions"] == "exact_phase7a_full_system"
    assert config["protocol"]["phase6_interventions_enabled"] is False
    mulcas = yaml.safe_load(config["paper_mulcas_config"].read_text(encoding="utf-8"))
    assert mulcas["complexity_control_n"] == 2
    assert mulcas["voting_threshold"] == 0
    assert [
        (viewer["negative_threshold"], viewer["positive_threshold"])
        for viewer in mulcas["viewer_sequence"][:2]
    ] == [(0.05, 0.95), (0.05, 0.51)]


def test_confusion_classification_accounts_for_each_row_once() -> None:
    outcomes = classify_outcomes([0, 0, 1, 1], [0, 1, 0, 1])
    assert outcomes.tolist() == [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]


def test_boolean_csv_values_are_parsed_deterministically() -> None:
    parsed = parse_boolean_series(pd.Series(["False", "True", "0", "1"]))
    assert parsed.tolist() == [False, True, False, True]


def test_clone_and_creator_context_use_strict_history_only() -> None:
    inventory, assigned = _context_frames()
    enriched = add_historical_context(inventory, assigned)
    p2 = enriched[enriched["period"] == "P2"].iloc[0]
    p3 = enriched[enriched["period"] == "P3"].iloc[0]
    assert p2["previously_seen_creator"]
    assert not p2["previously_ponzi_creator"]
    assert p2["historical_opcode_match_count"] == 1
    assert p3["historical_opcode_match_count"] == 2
    changed_future = assigned.copy()
    changed_future.loc[changed_future["period"] == "P3", ["opcode", "contractCode", "creationCode"]] = "changed"
    repeated = add_historical_context(inventory.iloc[[0]], changed_future)
    assert repeated.iloc[0]["historical_opcode_match_count"] == p2["historical_opcode_match_count"]


def test_contract_diagnostics_are_local_label_independent_and_finite() -> None:
    frame = pd.DataFrame(
        {
            "opcode": ["PUSH1 ADD SSTORE", "CALL STOP"],
            "contractCode": ["0x60010155", "0xf100"],
            "label": [0, 1],
        }
    )
    names_s = ["opcode_count", "opcode_entropy", "category_storage_ratio"]
    names_o = ["parseable_operand_count", "operand_width_mean"]
    first = add_contract_diagnostics(frame, names_s, names_o, small_constant_max=255)
    second = add_contract_diagnostics(
        frame.assign(label=1 - frame["label"]), names_s, names_o, small_constant_max=255
    )
    columns = names_s + names_o
    assert np.array_equal(first[columns].to_numpy(), second[columns].to_numpy())
    assert np.isfinite(first[columns].to_numpy(dtype=float)).all()
    changed_other = frame.copy()
    changed_other.loc[1, ["opcode", "contractCode"]] = ["INVALID", "0xfe"]
    third = add_contract_diagnostics(changed_other, names_s, names_o, small_constant_max=255)
    assert np.array_equal(first.loc[0, columns].to_numpy(), third.loc[0, columns].to_numpy())


def test_generated_inventory_exactly_matches_phase7a_confusion_counts() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    inventory = pd.read_csv(config["output_dir"] / "phase7d_error_inventory.csv")
    phase7a = json.loads(config["phase7a_results"].read_text(encoding="utf-8"))
    assert len(inventory) == sum(item["metrics"]["support_negative"] + item["metrics"]["support_positive"] for item in phase7a["scenarios"])
    assert inventory["row_index"].is_unique
    key_map = {"true_negative": "tn", "false_positive": "fp", "false_negative": "fn", "true_positive": "tp"}
    for scenario in phase7a["scenarios"]:
        period = scenario["prediction_period"]
        counts = inventory.loc[inventory["period"] == period, "error_type"].value_counts()
        for error_type, key in key_map.items():
            assert int(counts.get(error_type, 0)) == scenario["metrics"][key]


def test_inventory_contains_no_raw_code_and_has_required_diagnostics() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    inventory = pd.read_csv(config["output_dir"] / "phase7d_error_inventory.csv", nrows=2)
    assert not {"opcode", "contractCode", "creationCode"}.intersection(inventory.columns)
    required = {
        "row_index", "address", "creator", "error_type", "creator_history_feature",
        "previously_seen_creator", "previously_ponzi_creator", "opcode_count",
        "opcode_entropy", "parseable_operand_count", "operand_width_mean",
        "has_historical_opcode_clone", "has_historical_runtime_bytecode_clone",
        "routing_error_category", "near_boundary_wrong",
    }
    assert required.issubset(inventory.columns)


def test_prior_phase7_artifacts_and_reference_predictions_remain_unchanged() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert len(verify_frozen_manifest(config["frozen_manifest"])) == 10
    frozen = verify_prior_phase_chain(config["frozen_manifest"])
    assert "results/experiments/phase7a/phase7a_predictions.csv" in frozen
    assert "results/experiments/phase7b/phase7b_results.json" in frozen
    assert "results/experiments/phase7c/phase7c_results.json" in frozen
    assert "results/baseline/paper_aligned/mulcas_results.json" in frozen
    results = json.loads((config["output_dir"] / "phase7d_results.json").read_text(encoding="utf-8"))
    assert sha256_file(config["phase7a_predictions"]) == results["phase7a_predictions_sha256"]
    assert results["model_training_performed"] is False
    assert results["optimization_performed"] is False


def test_generated_outputs_are_deterministically_ordered() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    inventory = pd.read_csv(config["output_dir"] / "phase7d_error_inventory.csv")
    ordered = inventory.sort_values(
        ["period", "createdBlockNumber", "row_index"], kind="mergesort"
    ).reset_index(drop=True)
    assert inventory[["period", "createdBlockNumber", "row_index"]].equals(
        ordered[["period", "createdBlockNumber", "row_index"]]
    )
    pd.testing.assert_frame_equal(
        build_period_summary(inventory), build_period_summary(inventory)
    )
    pd.testing.assert_frame_equal(
        build_routing_summary(inventory), build_routing_summary(inventory)
    )
    pd.testing.assert_frame_equal(
        build_clone_summary(inventory), build_clone_summary(inventory)
    )
