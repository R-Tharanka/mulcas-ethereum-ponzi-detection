"""Phase 9C prospective-state, leakage, metric, and governance tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.extension.sealed_track_b import load_sealed_labels
from src.phase9.family_similarity import build_family_state
from src.phase9.phase9_metrics import paired_bootstrap, pooled_metrics
from src.phase9.phase9c import CANDIDATES, _advancement, load_config
from src.phase9.state_features import build_creator_state
from src.utils.paths import REPO_ROOT


CONFIG = REPO_ROOT / "configs/phase9/phase9c.yaml"
OUTPUT = REPO_ROOT / "results/phase9/phase9c"


def _frame(rows):
    frame = pd.DataFrame(rows)
    frame["opcode"] = frame.get("opcode", "PUSH1 STOP")
    return frame


def test_successor_test_anchor_migration_is_exact_and_documented():
    record = json.loads((REPO_ROOT / "results/extension/phase8r6d/test_anchor_migration.json").read_text())
    assert record["authorized_by_successor_phase"] == "phase8r6d"
    assert {item["test_file"] for item in record["changes"]} == {
        "tests/test_phase8r6b.py", "tests/test_phase8r6c.py"
    }
    for item in record["changes"]:
        path = REPO_ROOT / item["test_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["new_sha256"]
        assert "must not exist" not in " ".join(item["new_assertions"])


def test_sealed_track_b_labels_are_default_deny_during_phase9c():
    path = REPO_ROOT / "data/external/track_b_sealed/labels.csv"
    assert path.is_file()
    with pytest.raises(PermissionError, match="explicit final_evaluation=True is required"):
        load_sealed_labels(path)


def test_phase9c_source_never_loads_sealed_labels_or_p5_predictions():
    source = (REPO_ROOT / "src/phase9/phase9c.py").read_text(encoding="utf-8")
    assert "load_sealed_labels" not in source
    assert "track_b_sealed/labels.csv" not in source
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["track_b"]["access_permission"] == "disabled"
    assert config["temporal_protocol"]["p5_evaluation"] == "prohibited"
    assert [row["prediction_period"] for row in config["temporal_protocol"]["scenarios"]] == ["P2", "P3", "P4"]


def test_creator_state_uses_strictly_earlier_blocks_and_same_block_isolated():
    train = _frame([
        {"row_index": 0, "creator": "a", "createdBlockNumber": 10, "label": 1},
        {"row_index": 1, "creator": "a", "createdBlockNumber": 10, "label": 0},
        {"row_index": 2, "creator": "a", "createdBlockNumber": 20, "label": 0},
    ])
    target = _frame([
        {"row_index": 3, "creator": "a", "createdBlockNumber": 30},
        {"row_index": 4, "creator": "b", "createdBlockNumber": 30},
        {"row_index": 5, "creator": "b", "createdBlockNumber": 40},
    ])
    result = build_creator_state(train, target, sentinel=999)
    assert result.train_raw.prior_contract_count.tolist() == [0, 0, 2]
    assert result.train_raw.prior_verified_ponzi_count.tolist() == [0, 0, 1]
    assert result.target_raw.prior_contract_count.tolist() == [3, 0, 1]
    assert result.target_raw.prior_verified_ponzi_count.tolist() == [1, 0, 0]
    assert result.target_raw.blocks_since_creator_last_deployment.tolist() == [10, 999, 10]


def test_target_creator_labels_cannot_update_verified_state():
    train = _frame([{"row_index": 0, "creator": "a", "createdBlockNumber": 10, "label": 0}])
    target = _frame([
        {"row_index": 1, "creator": "b", "createdBlockNumber": 20},
        {"row_index": 2, "creator": "b", "createdBlockNumber": 30},
    ])
    result = build_creator_state(train, target, sentinel=999)
    assert result.target_raw.prior_verified_ponzi_count.tolist() == [0, 0]
    assert result.target_raw.prior_contract_count.tolist() == [0, 1]


def test_family_state_has_training_only_labels_and_prequential_unlabeled_counts():
    train = _frame([
        {"row_index": 0, "creator": "a", "createdBlockNumber": 10, "label": 1, "opcode": "PUSH1 STOP"},
        {"row_index": 1, "creator": "b", "createdBlockNumber": 10, "label": 0, "opcode": "PUSH1 STOP"},
        {"row_index": 2, "creator": "c", "createdBlockNumber": 20, "label": 0, "opcode": "PUSH1 STOP"},
    ])
    target = _frame([
        {"row_index": 3, "creator": "d", "createdBlockNumber": 30, "opcode": "PUSH1 STOP"},
        {"row_index": 4, "creator": "e", "createdBlockNumber": 40, "opcode": "PUSH1 STOP"},
    ])
    result = build_family_state(train, target, min_df=1, threshold=0.90)
    assert result.train_raw.loc[0, "count_prior_all_similarity_ge_0_90"] == 0
    assert result.train_raw.loc[1, "count_prior_all_similarity_ge_0_90"] == 0
    assert result.train_raw.loc[2, "count_prior_ponzi_similarity_ge_0_90"] == 1
    assert result.target_raw.loc[0, "count_prior_all_similarity_ge_0_90"] == 3
    assert result.target_raw.loc[1, "count_prior_all_similarity_ge_0_90"] == 4
    assert result.target_raw.loc[0, "count_prior_ponzi_similarity_ge_0_90"] == 1
    assert result.target_raw.loc[1, "count_prior_ponzi_similarity_ge_0_90"] == 1


def test_family_threshold_is_frozen():
    train = _frame([{"row_index": 0, "creator": "a", "createdBlockNumber": 10, "label": 0}])
    target = _frame([{"row_index": 1, "creator": "b", "createdBlockNumber": 20}])
    with pytest.raises(ValueError, match="must remain 0.90"):
        build_family_state(train, target, min_df=1, threshold=0.89)


def test_config_freezes_only_c0_to_c3_and_phase9b_rules():
    config = load_config(CONFIG)
    assert tuple(config["candidates"]) == CANDIDATES
    assert config["bootstrap"] == {"replicates": 2000, "seed": 42, "resampling_unit": "contract_row"}
    assert config["metrics"]["reference_fpr_ceiling"] == pytest.approx(0.049539594843462244)
    assert config["advancement"]["maximum_allowed_single_period_f1_degradation"] == 0.02


def test_frozen_mulcas_thresholds_and_voting_are_unchanged():
    values = yaml.safe_load((REPO_ROOT / "configs/paper_aligned_mulcas.yaml").read_text())
    assert values["voting_threshold"] == 0
    assert [(v.get("negative_threshold"), v.get("positive_threshold")) for v in values["viewer_sequence"][:2]] == [(0.05, 0.95), (0.05, 0.51)]


def test_phase9_metrics_and_bootstrap_are_deterministic():
    truth = np.array([0, 0, 0, 1, 1, 1])
    control = {"truth": truth, "prediction": np.array([0, 0, 1, 0, 1, 1]), "score": np.array([.1, .2, .7, .3, .8, .9])}
    candidate = {"truth": truth, "prediction": np.array([0, 0, 0, 0, 1, 1]), "score": np.array([.1, .2, .4, .3, .8, .9])}
    metrics = pooled_metrics(truth, candidate["prediction"], candidate["score"], fpr_ceiling=.05, reference_prevalence=.048322560788)
    assert metrics["fpr"] == 0
    first = paired_bootstrap(control, candidate, replicates=100, seed=42)
    second = paired_bootstrap(control, candidate, replicates=100, seed=42)
    assert first == second


def test_advancement_rule_requires_every_frozen_condition():
    config = load_config(CONFIG)
    pooled = pd.DataFrame([
        {"candidate": "C0", "f1": .4, "pr_auc": .5, "fpr": .04},
        {"candidate": "C1", "f1": .41, "pr_auc": .495, "fpr": .04},
        {"candidate": "C2", "f1": .42, "pr_auc": .48, "fpr": .04},
        {"candidate": "C3", "f1": .42, "pr_auc": .5, "fpr": .06},
    ])
    periods = pd.DataFrame([
        {"candidate": c, "period": p, "f1": value}
        for c, values in {"C0": [.1,.2,.8], "C1": [.1,.2,.8], "C2": [.1,.2,.8], "C3": [.1,.2,.8]}.items()
        for p, value in zip(("P2","P3","P4"), values)
    ])
    bootstrap = pd.DataFrame({"candidate": ["C1", "C2", "C3"]})
    decision = _advancement(pooled, periods, bootstrap, config).set_index("candidate")
    assert bool(decision.loc["C1", "advanced"])
    assert not bool(decision.loc["C2", "advanced"])
    assert not bool(decision.loc["C3", "advanced"])


def test_track_a_raw_hash_is_unchanged():
    path = REPO_ROOT / "data/raw/PonziDataset_20221114.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"


def test_phase9c_outputs_are_complete_and_track_b_free():
    required = {
        "phase9c_results.json", "runtime_preflight.json", "candidate_metrics.csv",
        "period_metrics.csv", "pooled_predictions.csv", "state_feature_summary.csv",
        "family_similarity_summary.csv", "bootstrap_comparisons.csv",
        "advancement_decisions.csv", "subgroup_metrics.csv", "changed_predictions.csv",
        "track_b_access_audit.json", "test_anchor_migration.json", "phase9c_report.md",
    }
    assert required.issubset({path.name for path in OUTPUT.iterdir()})
    result = json.loads((OUTPUT / "phase9c_results.json").read_text())
    audit = json.loads((OUTPUT / "track_b_access_audit.json").read_text())
    assert result["track_b_accessed"] is False and result["p5_evaluated"] is False
    assert audit == {
        "track_b_accessed": False, "sealed_labels_opened": False,
        "track_b_rows_fitted": 0, "track_b_scores_generated": 0,
        "track_b_metrics_generated": 0, "access_permission": "disabled",
        "p5_evaluated": False,
    }


def test_c0_reproduces_frozen_pooled_and_period_controls():
    result = json.loads((OUTPUT / "phase9c_results.json").read_text())
    assert result["control"]["pooled_tn"] == 5161
    assert result["control"]["pooled_fp"] == 269
    assert result["control"]["pooled_fn"] == 84
    assert result["control"]["pooled_tp"] == 66
    periods = pd.read_csv(OUTPUT / "period_metrics.csv")
    expected = {"P2": .08280254777070063, "P3": .2222222222222222, "P4": .8518518518518519}
    control = periods[periods.candidate.eq("C0")].set_index("period")
    for period, f1 in expected.items():
        assert control.loc[period, "f1"] == pytest.approx(f1, abs=1e-12)


def test_pooled_predictions_contain_only_track_a_development_candidates():
    predictions = pd.read_csv(OUTPUT / "pooled_predictions.csv")
    assert set(predictions.candidate) == set(CANDIDATES)
    assert set(predictions.period) == {"P2", "P3", "P4"}
    assert predictions.groupby("candidate").size().eq(5580).all()
    assert set(CREATOR for CREATOR in ["prior_contract_count", "prior_verified_ponzi_flag"]).issubset(predictions.columns)
    assert {"max_similarity_prior_ponzi", "count_prior_all_similarity_ge_0_90"}.issubset(predictions.columns)


def test_phase9c_decision_and_migration_copy_are_deterministic():
    result = json.loads((OUTPUT / "phase9c_results.json").read_text())
    decisions = pd.read_csv(OUTPUT / "advancement_decisions.csv")
    assert result["advanced_candidates"] == []
    assert result["selected_candidate"] is None
    assert result["h9c_supported"] is False
    assert not decisions.advanced.any()
    source = REPO_ROOT / "results/extension/phase8r6d/test_anchor_migration.json"
    copied = OUTPUT / "test_anchor_migration.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == hashlib.sha256(copied.read_bytes()).hexdigest()
