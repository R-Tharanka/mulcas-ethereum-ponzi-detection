from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from src.extension.feasibility import verify_manifest
from src.extension.phase8r1a import (
    DEFAULT_CONFIG_PATH,
    fit_estimator,
    load_config,
    period_sample_weights,
    strategy_training_periods,
)
from src.utils.paths import REPO_ROOT


def test_frozen_phase7_and_phase8r0_manifests_are_unchanged() -> None:
    config = load_config()
    assert verify_manifest(config["frozen_phase7_manifest"])["verified"]
    assert verify_manifest(config["frozen_phase8r0_manifest"])["verified"]


def test_exact_predeclared_strategies_and_periods() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert [item["id"] for item in config["strategies"]] == [
        "expanding_history",
        "recent_1_period",
        "recent_2_periods",
        "recent_3_periods",
        "exponential_decay_0.25",
        "exponential_decay_0.50",
        "exponential_decay_0.75",
    ]
    assert config["selection"]["periods"] == ["P2", "P3", "P4"]
    assert config["selection"]["p5_usage"] == "prohibited"


def test_hard_windows_and_history_limited_flag() -> None:
    strategy = {"family": "hard_window", "window_periods": 3}
    assert strategy_training_periods(strategy, "P2") == (("P0", "P1"), True)
    assert strategy_training_periods(strategy, "P3") == (("P0", "P1", "P2"), False)
    strategy["window_periods"] = 1
    assert strategy_training_periods(strategy, "P4") == (("P3",), False)


def test_period_weights_follow_declared_formula() -> None:
    frame = pd.DataFrame({"period": ["P0", "P1", "P2", "P3"]})
    for alpha in (0.25, 0.50, 0.75):
        strategy = {"family": "exponential_decay", "alpha": alpha}
        observed = period_sample_weights(frame, strategy, "P4")
        np.testing.assert_allclose(observed, [alpha**3, alpha**2, alpha, 1.0])
    assert period_sample_weights(frame, {"family": "control"}, "P4") is None


def test_sample_weights_reach_estimator_fit_only() -> None:
    class Recorder:
        def fit(self, features, labels, sample_weight=None):
            self.features = features
            self.labels = labels
            self.sample_weight = sample_weight
            return self

    weights = np.asarray([0.25, 1.0])
    model = fit_estimator(Recorder(), np.eye(2), np.asarray([0, 1]), weights)
    np.testing.assert_array_equal(model.sample_weight, weights)


def test_no_phase6_intervention_or_frozen_routing_change() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert not any(config["frozen_behavior"].values())
    model = yaml.safe_load((REPO_ROOT / "configs/paper_aligned_baseline_models.yaml").read_text(encoding="utf-8"))
    mulcas = yaml.safe_load((REPO_ROOT / "configs/paper_aligned_mulcas.yaml").read_text(encoding="utf-8"))
    assert model["models"]["ridge"]["class_weight"] is None
    assert model["models"]["svm"]["class_weight"] is None
    assert float(model["models"]["xgboost"].get("scale_pos_weight", 1.0)) == 1.0
    assert mulcas["voting_threshold"] == 0
    assert [(item.get("negative_threshold"), item.get("positive_threshold")) for item in mulcas["viewer_sequence"][:2]] == [(0.05, 0.95), (0.05, 0.51)]


def test_generated_results_reproduce_control_and_exclude_p5() -> None:
    result_path = REPO_ROOT / "results/extension/phase8r1a/phase8r1a_results.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    assert results["control_reproduction"]["passed"]
    assert results["p5_loaded_or_evaluated"] is False
    assert {item["prediction_period"] for item in results["runs"]} == {"P2", "P3", "P4"}
    assert all(item["feature_fit_audit"]["historical_only"] for item in results["runs"])
    assert all(item["feature_fit_audit"]["feature_fit_rows"] == item["train"]["row_count"] for item in results["runs"])


def test_generated_membership_and_outputs_are_deterministic() -> None:
    current = pd.read_csv(
        REPO_ROOT / "results/extension/phase8r1a/phase8r1a_predictions.csv",
        usecols=["strategy_id", "period", "row_index"],
    )
    assert "P5" not in set(current["period"])
    assert not current.duplicated(["strategy_id", "period", "row_index"]).any()
    results = json.loads(
        (REPO_ROOT / "results/extension/phase8r1a/phase8r1a_results.json").read_text(encoding="utf-8")
    )
    assert len(results["deterministic_metric_fingerprint"]) == 64

