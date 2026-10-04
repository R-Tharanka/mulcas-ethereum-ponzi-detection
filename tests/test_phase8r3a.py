from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.extension.phase8r3a import (
    DISAGREEMENT_CATEGORIES,
    align_predictions,
    disagreement_summary,
    hybrid_decision,
    load_config,
    negative_complementarity,
    novelty_creator_interaction,
    oracle_headroom,
    positive_complementarity,
    verify_frozen_inputs,
)
from src.utils.paths import REPO_ROOT


def test_all_frozen_inputs_verify_and_protocol_remains_frozen() -> None:
    config = load_config()
    assert len(verify_frozen_inputs(config)) >= 73
    policy = config["performance_preflight_policy"].read_text(encoding="utf-8")
    assert "30 minutes" in policy
    assert "explicit user approval" in policy


def test_alignment_is_exact_p2_to_p4_only() -> None:
    aligned = align_predictions(load_config())
    assert len(aligned) == 5580
    assert aligned["row_index"].nunique() == 5580
    assert aligned.groupby("period").size().to_dict() == {"P2": 5141, "P3": 289, "P4": 150}
    assert "P5" not in set(aligned["period"])


def test_ensemble_is_arithmetic_mean_with_native_threshold() -> None:
    config = load_config()
    aligned = align_predictions(config)
    columns = [f"gin_probability_seed_{seed}" for seed in config["inputs"]["gin_seeds"]]
    expected = aligned[columns].mean(axis=1)
    np.testing.assert_allclose(aligned["gin_mean_probability"], expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(aligned["gin_ensemble_prediction"], expected.ge(0.5).astype(int))


def test_every_row_has_one_reconciling_disagreement_category() -> None:
    config = load_config()
    aligned = align_predictions(config)
    assert set(aligned["disagreement_category"]) == set(DISAGREEMENT_CATEGORIES)
    summary = disagreement_summary(aligned, config["inputs"]["gin_seeds"])
    ensemble = summary[summary["gin_policy"].eq("five_seed_mean")]
    assert set(summary["gin_policy"]) == {"five_seed_mean", "seed_7", "seed_42", "seed_123", "seed_2026", "seed_999"}
    expected = {"P2": 5141, "P3": 289, "P4": 150, "P2-P4": 5580}
    assert ensemble.groupby("period")["count"].sum().to_dict() == expected
    interaction = novelty_creator_interaction(aligned)
    assert interaction.groupby("period")["row_count"].sum().to_dict() == expected
    assert interaction.groupby("period")["true_ponzi_count"].sum().to_dict() == {
        "P2": 50, "P3": 50, "P4": 50, "P2-P4": 150,
    }


def test_positive_negative_and_oracle_counts_reconcile() -> None:
    config = load_config()
    aligned = align_predictions(config)
    seeds = config["inputs"]["gin_seeds"]
    positive = positive_complementarity(aligned, seeds)
    negative = negative_complementarity(aligned, seeds)
    for row in positive.itertuples(index=False):
        assert row.tp_by_both + row.tp_only_mulcas + row.tp_only_gin + row.fn_by_both == row.positive_support
        assert bool(row.oracle_non_deployable)
    for row in negative.itertuples(index=False):
        assert row.tn_by_both + row.tn_only_mulcas + row.tn_only_gin + row.fp_by_both == row.negative_support
    oracle = oracle_headroom(aligned, seeds)
    assert oracle["non_deployable"].astype(bool).all()
    assert (oracle[["tn", "fp", "fn", "tp"]].sum(axis=1) == oracle["support_negative"] + oracle["support_positive"]).all()


def test_hybrid_gate_is_predeclared_and_deterministic() -> None:
    config = load_config()
    aligned_a = align_predictions(config)
    aligned_b = align_predictions(config)
    pd.testing.assert_frame_equal(aligned_a, aligned_b)
    positive = positive_complementarity(aligned_a, config["inputs"]["gin_seeds"])
    first = hybrid_decision(aligned_a, positive, config)
    second = hybrid_decision(aligned_b, positive, config)
    assert first == second
    assert first["hybrid_experiment_justified"] is True
    assert all(first["predeclared_conditions"].values())


def test_generated_outputs_record_analysis_only_guards() -> None:
    result = json.loads(
        (REPO_ROOT / "results/extension/phase8r3a/phase8r3a_results.json").read_text(encoding="utf-8")
    )
    assert result["models_trained"] is False
    assert result["threshold_tuned"] is False
    assert result["p5_loaded_or_evaluated"] is False
    assert result["oracle_results_non_deployable"] is True
    source = (REPO_ROOT / "src/extension/phase8r3a.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "LogisticRegression" not in source
    assert "torch" not in source
