from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.static_grouped_improvement import (
    ap_weights,
    clone_cap,
    load_config,
    select_candidate,
)


def test_candidate_set_and_test_policy_are_frozen():
    config = load_config()
    assert config["candidates"] == [
        "C0_full_equal",
        "C1_clone_capped_equal",
        "C2_full_ap_weighted",
        "C3_clone_capped_ap_weighted",
    ]
    assert config["test_policy"]["score_only_selected_candidate"] is True
    assert config["test_policy"]["no_test_reselection"] is True


def test_clone_cap_keeps_one_row_per_group_label_pair():
    frame = pd.DataFrame(
        {
            "row_index": [5, 2, 7, 8],
            "duplicate_group": ["A", "A", "A", "B"],
            "label": [0, 0, 1, 0],
        }
    )
    capped = clone_cap(frame)
    assert list(capped.row_index) == [2, 7, 8]
    assert not capped.duplicated(["duplicate_group", "label"]).any()


def test_ap_weights_are_positive_and_normalized():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.7], [0.9, 0.8]])
    weights = ap_weights(labels, scores)
    assert np.all(weights > 0)
    assert np.isclose(weights.sum(), 1.0)


def test_selection_uses_validation_metrics_and_frozen_tie_order():
    validation = pd.DataFrame({"label": [0, 0, 1, 1]})
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    score_sets = {name: scores.copy() for name in (
        "C0_full_equal", "C1_clone_capped_equal",
        "C2_full_ap_weighted", "C3_clone_capped_ap_weighted",
    )}
    weights = {name: [1 / 3] * 3 for name in score_sets}
    _, selected = select_candidate(validation, score_sets, weights)
    assert selected["candidate"] == "C0_full_equal"


def test_guards_prohibit_p5_track_b_and_hyperparameter_search():
    config = load_config()
    assert config["guards"] == {
        "p5_loaded": False,
        "track_b_loaded": False,
        "historical_results_modified": False,
        "outer_group_split_changed": False,
    }
    assert config["model_policy"]["new_hyperparameter_search"] is False
