"""Research-governance and reconciliation tests for Phase 9A."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.phase6a import verify_frozen_manifest
from src.phase9.creator_oracle import online_true_label_creator_flags
from src.phase9.protocol_gap import load_config, verify_inputs
from src.utils.paths import REPO_ROOT


OUTPUT = REPO_ROOT / "results" / "phase9" / "phase9a"


def _results() -> dict:
    return json.loads((OUTPUT / "phase9a_results.json").read_text(encoding="utf-8"))


def test_phase8_closure_and_full_chain_remain_frozen() -> None:
    config = load_config()
    assert len(verify_frozen_manifest(config["phase8_closure_manifest"])) == 11
    verification = verify_inputs(config)
    assert verification["verified"] is True
    assert verification["phase0_to_phase8r5c1_anchor_count"] == 199
    assert verification["raw_dataset_sha256"] == "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"


def test_frozen_p5_controls_match_archived_results() -> None:
    controls = _results()["p5_controls"]
    assert controls["phase55"] == {
        "precision": 0.90625, "recall": 0.453125,
        "f1": 0.6041666666666666, "tn": 442, "fp": 3, "fn": 35, "tp": 29,
    }
    assert controls["phase7a"] == {
        "precision": 0.8709677419354839, "recall": 0.421875,
        "f1": 0.5684210526315789, "tn": 441, "fp": 4, "fn": 37, "tp": 27,
    }


def test_creator_oracle_delays_same_block_label_updates() -> None:
    train = pd.DataFrame({"creator": ["known"], "label": [1]})
    target = pd.DataFrame(
        {
            "row_index": [1, 2, 3],
            "createdBlockNumber": [10, 10, 11],
            "creator": ["new", "new", "new"],
            "label": [1, 0, 0],
        }
    )
    assert np.array_equal(online_true_label_creator_flags(train, target), [0.0, 0.0, 1.0])


def test_creator_oracle_is_explicitly_non_deployable_and_replay_exact() -> None:
    results = _results()
    assert results["creator_oracle_summary"]["all_frozen_replays_exact"] is True
    metrics = pd.read_csv(OUTPUT / "creator_oracle_period_metrics.csv")
    assert set(metrics["period"]) == {"P2", "P3", "P4", "P5"}
    assert set(metrics.loc[metrics["condition"] == "oracle", "deployability"]) == {
        "non_deployable_true_label_online_oracle"
    }


def test_viewer_selection_partial_results_are_not_ranked() -> None:
    results = _results()
    assert results["viewer_selection_oracle"]["candidate_count"] == 0
    selection = pd.read_csv(OUTPUT / "viewer_selection_oracle.csv")
    assert selection.loc[0, "status"] == "stopped_by_runtime_gate"
    assert not bool(selection.loc[0, "selection_complete"])
    assert not bool(selection.loc[0, "partial_ranking_reported"])


def test_combined_oracle_and_gap_fraction_are_unavailable() -> None:
    results = _results()
    assert results["combined_oracle"]["status"] == "unavailable"
    gap = pd.read_csv(OUTPUT / "gap_decomposition.csv")
    assert gap["D4"].isna().all()
    assert gap["gap_closed_fraction"].isna().all()
    assert set(gap["status"]) == {"N/A_D4_unavailable"}


def test_ngram_audit_is_dimension_only() -> None:
    audit = pd.read_csv(OUTPUT / "ngram_dimension_audit.csv")
    assert len(audit) == 6
    assert not audit["classifier_trained"].astype(bool).any()
    assert set(audit["corpus"]) == {"P0-P3", "P0-P4"}


def test_p2_false_positive_inventory_reconciles() -> None:
    rows = pd.read_csv(OUTPUT / "p2_fp_routing.csv")
    assert len(rows) == 251
    assert set(rows["period"]) == {"P2"}
    assert set(rows["error_type"]) == {"false_positive"}
    assert rows["normalized_opcode_sha256"].str.len().eq(64).all()
    assert rows["normalized_runtime_sha256"].str.len().eq(64).all()


def test_p2_membership_and_track_a_governance() -> None:
    membership = pd.read_csv(OUTPUT / "p2_membership_audit.csv")
    positive = membership.loc[membership["check"] == "positive_count"].iloc[0]
    assert positive["status"] == "MATCH"
    assert int(positive["observed"]) == 50
    results = _results()
    assert results["scope"]["p5_status"] == "previously_observed_exploratory_track_a"
    assert results["scope"]["phase9b_started"] is False
    assert results["scope"]["new_classifier"] is False
    assert results["scope"]["threshold_changes"] is False


def test_all_required_outputs_exist() -> None:
    required = {
        "phase9a_results.json", "protocol_evidence.csv", "p5_decomposition.csv",
        "creator_oracle_period_metrics.csv", "creator_oracle_changed_rows.csv",
        "viewer_candidate_metrics.csv", "viewer_selection_oracle.csv",
        "combined_oracle_metrics.csv", "ngram_dimension_audit.csv",
        "p2_fp_routing.csv", "p2_fp_groups.csv", "p2_membership_audit.csv",
        "gap_decomposition.csv", "runtime_preflight.json", "phase9a_report.md",
    }
    assert {path.name for path in OUTPUT.iterdir()} == required
