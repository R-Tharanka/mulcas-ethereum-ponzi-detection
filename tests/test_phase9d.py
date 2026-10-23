"""Phase 9D calibration, routing, temporal, and governance tests."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.phase9.phase9d import (
    CANDIDATES,
    _advancement,
    _clip_logit,
    _guard,
    _project_bootstrap_seconds,
    _route,
    load_config,
)
from src.utils.paths import REPO_ROOT


CONFIG = REPO_ROOT / "configs/phase9/phase9d.yaml"
OUTPUT = REPO_ROOT / "results/phase9/phase9d"


def test_phase9d_config_is_prospective_and_fixed():
    config = load_config(CONFIG)
    assert tuple(config["candidates"]) == CANDIDATES
    assert config["routing"]["positive_confirmation_votes"] == 2
    assert config["routing"]["negative_recovery_votes"] == 3
    assert config["routing"]["threshold_search"] == "prohibited"
    assert config["calibration"]["temporal_pairs"] == {
        "P2": {"fit_periods": ["P0"], "calibration_period": "P1"},
        "P3": {"fit_periods": ["P0", "P1"], "calibration_period": "P2"},
        "P4": {"fit_periods": ["P0", "P1", "P2"], "calibration_period": "P3"},
    }


def test_track_b_and_p5_are_default_denied_by_phase9d():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source = (REPO_ROOT / "src/phase9/phase9d.py").read_text(encoding="utf-8")
    assert config["track_b"]["access_permission"] == "disabled"
    assert config["temporal_protocol"]["p5_evaluation"] == "prohibited"
    assert "load_sealed_labels" not in source
    assert "track_b_sealed" not in source
    assert '"P5"' not in source


def test_probability_logit_clipping_is_finite_and_deterministic():
    first = _clip_logit(np.array([0.0, 0.5, 1.0]), 1e-6)
    second = _clip_logit(np.array([0.0, 0.5, 1.0]), 1e-6)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert first[1, 0] == pytest.approx(0.0)


def test_bootstrap_runtime_projection_accounts_for_all_candidates():
    assert _project_bootstrap_seconds(2.0, 20, 3, 2000) == pytest.approx(600.0)
    with pytest.raises(ValueError, match="must be positive"):
        _project_bootstrap_seconds(2.0, 0, 3, 2000)


def test_frozen_route_thresholds_and_voting_are_preserved():
    mulcas = yaml.safe_load((REPO_ROOT / "configs/paper_aligned_mulcas.yaml").read_text())
    scores = {
        ("cascade", "xgboost_term_count"): np.array([0.01, 0.50, 0.99]),
        ("cascade", "xgboost_ngram_count_df_filtered"): np.array([0.50, 0.01, 0.50]),
        ("voting", "xgboost_term_count"): np.array([0.1, 0.1, 0.1]),
        ("voting", "xgboost_ngram_count_df_filtered"): np.array([0.1, 0.1, 0.1]),
        ("voting", "xgboost_ngram_tfidf_df_filtered"): np.array([0.1, 0.1, 0.1]),
        ("voting", "ridge_tfidf"): np.array([0.1, 0.6, 0.1]),
    }
    prediction, _, route, votes, _ = _route(scores, mulcas)
    assert prediction.tolist() == [0, 0, 1]
    assert route.tolist() == ["xgboost_term_count", "xgboost_ngram_count_df_filtered", "xgboost_term_count"]
    assert votes.tolist() == [0, 1, 0]


def test_fixed_guard_confirms_two_votes_and_requires_three_for_recovery():
    config = load_config(CONFIG)
    base = np.array([1, 1, 0, 0])
    votes = np.array([2, 1, 3, 2])
    matrix = np.array([
        [.9, .8, .1, .1], [.9, .1, .1, .1],
        [.9, .8, .7, .1], [.9, .8, .1, .1],
    ])
    prediction, score, route = _guard(base, votes, matrix, config)
    assert prediction.tolist() == [1, 0, 1, 0]
    assert score.tolist() == pytest.approx(matrix.mean(axis=1).tolist())
    assert route.tolist() == [
        "guard_confirmed", "guard_rejected_positive",
        "guard_recovered_negative", "guard_confirmed",
    ]


def test_phase9b_advancement_rule_is_not_weakened():
    config = load_config(CONFIG)
    pooled = pd.DataFrame([
        {"candidate": "D0", "f1": .30, "pr_auc": .40, "fpr": .049},
        {"candidate": "D1", "f1": .31, "pr_auc": .395, "fpr": .048},
        {"candidate": "D2", "f1": .31, "pr_auc": .38, "fpr": .048},
        {"candidate": "D3", "f1": .31, "pr_auc": .40, "fpr": .06},
    ])
    periods = pd.DataFrame([
        {"candidate": candidate, "period": period, "f1": value}
        for candidate, values in {
            "D0": [.1, .2, .8], "D1": [.1, .2, .8],
            "D2": [.1, .2, .8], "D3": [.1, .2, .8],
        }.items()
        for period, value in zip(("P2", "P3", "P4"), values)
    ])
    bootstrap = pd.DataFrame({"candidate": ["D1", "D2", "D3"]})
    decision = _advancement(pooled, periods, bootstrap, config).set_index("candidate")
    assert bool(decision.loc["D1", "advanced"])
    assert not bool(decision.loc["D2", "advanced"])
    assert not bool(decision.loc["D3", "advanced"])


def test_track_a_raw_hash_is_unchanged():
    path = REPO_ROOT / "data/raw/PonziDataset_20221114.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"


def test_phase9d_outputs_are_complete_when_present():
    if not OUTPUT.is_dir():
        pytest.skip("Phase 9D has not been executed yet.")
    required = {
        "phase9d_results.json", "runtime_preflight.json", "candidate_metrics.csv",
        "period_metrics.csv", "pooled_predictions.csv", "calibration_audit.csv",
        "routing_summary.csv", "bootstrap_comparisons.csv", "advancement_decisions.csv",
        "changed_predictions.csv", "track_b_access_audit.json", "phase9d_report.md",
    }
    assert required.issubset({path.name for path in OUTPUT.iterdir()})
    result = json.loads((OUTPUT / "phase9d_results.json").read_text())
    assert result["track_b_accessed"] is False
    assert result["p5_evaluated"] is False
    assert result["phase9e_started"] is False


def test_phase9d_control_reproduces_frozen_forward_counts_when_present():
    result_path = OUTPUT / "phase9d_results.json"
    if not result_path.is_file():
        pytest.skip("Phase 9D has not been executed yet.")
    control = json.loads(result_path.read_text())["control"]
    assert (control["pooled_tn"], control["pooled_fp"], control["pooled_fn"], control["pooled_tp"]) == (5161, 269, 84, 66)
