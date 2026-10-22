from src.experiments.static_grouped_sparse_svm import load_config


def test_sparse_svm_is_single_fixed_candidate():
    config = load_config()
    assert config["model"] == {"type": "LinearSVC", "C": 1.0, "class_weight": "balanced"}
    assert config["hyperparameter_search"] is False


def test_sparse_svm_preserves_group_and_external_guards():
    guards = load_config()["guards"]
    assert guards["exact_group_safe_split"] is True
    assert guards["p5_loaded"] is False
    assert guards["track_b_loaded"] is False
    assert guards["parent_modified"] is False


def test_sparse_svm_threshold_is_validation_only():
    config = load_config()
    assert config["threshold"] == "validation_f1_selected"
    assert config["test_policy"] == "one_confirmation_only"
