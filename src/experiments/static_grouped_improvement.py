"""Controlled clone-balanced improvement of the grouped random diagnostic."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from scripts.static_random_mulcas import best_f1_threshold, evaluate, make_xgb_pipeline
from src.experiments.static_grouped_mulcas import (
    DATASET,
    EXPECTED_DATASET_SHA256,
    EXPECTED_HISTORICAL_SHA256,
    HISTORICAL_RESULT,
    ROOT,
    RESULT_DIR as PARENT_DIR,
    grouped_split,
    load_grouped_data,
    make_grouped_ridge,
    sha256_file,
)


CONFIG = ROOT / "configs" / "experiments" / "static_grouped_improvement.yaml"
OUTPUT_DIR = ROOT / "results" / "experiments" / "static_grouped_improvement"
PARENT_RESULT = PARENT_DIR / "static_grouped_mulcas_results.json"


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def clone_cap(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one deterministic row for each group-label pair."""
    return (
        frame.sort_values("row_index")
        .drop_duplicates(["duplicate_group", "label"], keep="first")
        .reset_index(drop=True)
    )


def fit_viewers(frame: pd.DataFrame, parent: dict, seed: int, max_features: int, workers: int):
    count = make_xgb_pipeline("count", seed, max_features, workers)
    count.set_params(**parent["models"]["xgboost_count"]["best_params"])
    count.fit(frame.opcode, frame.label.to_numpy())

    tfidf = make_xgb_pipeline("tfidf", seed, max_features, workers)
    tfidf.set_params(**parent["models"]["xgboost_tfidf"]["best_params"])
    tfidf.fit(frame.opcode, frame.label.to_numpy())

    ridge = make_grouped_ridge(frame, seed, max_features)
    ridge.fit(frame.opcode, frame.label.to_numpy())
    return [count, tfidf, ridge]


def viewer_scores(models, documents) -> np.ndarray:
    return np.column_stack([model.predict_proba(documents)[:, 1] for model in models])


def ap_weights(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(
        [average_precision_score(labels, scores[:, index]) for index in range(scores.shape[1])],
        dtype=float,
    )
    return values / values.sum()


def candidate_scores(scores: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return scores.mean(axis=1)
    return scores @ weights


def select_candidate(validation: pd.DataFrame, score_sets: dict[str, np.ndarray], weights: dict[str, list[float]]):
    labels = validation.label.to_numpy()
    order = [
        "C0_full_equal",
        "C1_clone_capped_equal",
        "C2_full_ap_weighted",
        "C3_clone_capped_ap_weighted",
    ]
    rows = []
    for candidate in order:
        scores = score_sets[candidate]
        threshold, f1 = best_f1_threshold(labels, scores)
        rows.append(
            {
                "candidate": candidate,
                "validation_f1": f1,
                "validation_average_precision": float(average_precision_score(labels, scores)),
                "threshold": threshold,
                "viewer_weights": weights[candidate],
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["validation_f1"],
            row["validation_average_precision"],
            -order.index(row["candidate"]),
        ),
    )
    return rows, selected


def verify_inputs(config: dict) -> dict:
    observed = {
        "dataset": sha256_file(DATASET),
        "historical_result": sha256_file(HISTORICAL_RESULT),
        "parent_result": sha256_file(PARENT_RESULT),
    }
    expected = {
        "dataset": config["dataset_sha256"],
        "historical_result": config["historical_result_sha256"],
        "parent_result": config["parent_result_sha256"],
    }
    if observed != expected:
        raise RuntimeError(f"Frozen input mismatch: expected={expected}, observed={observed}")
    return observed


def run(config_path: Path = CONFIG, output_dir: Path = OUTPUT_DIR) -> dict:
    config = load_config(config_path)
    hashes_before = verify_inputs(config)
    started = time.perf_counter()
    parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    frame = load_grouped_data(DATASET)
    partitions = grouped_split(frame, int(config["seed"]))
    fit = partitions["fit"]
    validation = partitions["validation"]
    test = partitions["test"]
    capped = clone_cap(fit)

    full_models = fit_viewers(
        fit, parent, int(config["seed"]), int(config["max_features"]), int(config["workers"])
    )
    capped_models = fit_viewers(
        capped, parent, int(config["seed"]), int(config["max_features"]), int(config["workers"])
    )
    y_validation = validation.label.to_numpy()
    full_validation = viewer_scores(full_models, validation.opcode)
    capped_validation = viewer_scores(capped_models, validation.opcode)
    full_weights = ap_weights(y_validation, full_validation)
    capped_weights = ap_weights(y_validation, capped_validation)

    score_sets = {
        "C0_full_equal": candidate_scores(full_validation, None),
        "C1_clone_capped_equal": candidate_scores(capped_validation, None),
        "C2_full_ap_weighted": candidate_scores(full_validation, full_weights),
        "C3_clone_capped_ap_weighted": candidate_scores(capped_validation, capped_weights),
    }
    weights = {
        "C0_full_equal": [1 / 3] * 3,
        "C1_clone_capped_equal": [1 / 3] * 3,
        "C2_full_ap_weighted": full_weights.tolist(),
        "C3_clone_capped_ap_weighted": capped_weights.tolist(),
    }
    validation_rows, selected = select_candidate(validation, score_sets, weights)

    selected_name = selected["candidate"]
    selected_models = capped_models if "clone_capped" in selected_name else full_models
    selected_weights = None if selected_name.endswith("equal") else np.asarray(selected["viewer_weights"])
    test_matrix = viewer_scores(selected_models, test.opcode)
    test_scores = candidate_scores(test_matrix, selected_weights)
    test_metrics = evaluate(test.label.to_numpy(), test_scores, float(selected["threshold"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(output_dir / "validation_candidates.csv", index=False)
    predictions = test[["row_index", "address", "label", "duplicate_group", "opcode_sha256"]].copy()
    predictions["score"] = test_scores
    predictions["prediction"] = (test_scores >= float(selected["threshold"])).astype(int)
    predictions["candidate"] = selected_name
    predictions.to_csv(output_dir / "selected_test_predictions.csv", index=False)

    parent_test = parent["test"]
    result = {
        "status": "complete",
        "experiment": "clone_balanced_group_safe_static_ensemble",
        "frozen_inputs": hashes_before,
        "dataset_rows": len(frame),
        "fit_rows": len(fit),
        "clone_capped_fit_rows": len(capped),
        "fit_rows_removed_by_cap": len(fit) - len(capped),
        "validation_candidates": validation_rows,
        "selected_candidate": selected_name,
        "selection_data": "validation_only",
        "selected_test": test_metrics,
        "parent_group_safe_test": parent_test,
        "selected_minus_parent": {
            key: float(test_metrics[key] - parent_test[key])
            for key in ("precision", "recall", "f1", "average_precision")
        },
        "guards": {
            "p5_loaded": False,
            "track_b_loaded": False,
            "test_used_for_candidate_selection": False,
            "outer_group_split_changed": False,
            "new_hyperparameter_search": False,
            "historical_results_modified": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    hashes_after = verify_inputs(config)
    result["frozen_inputs_preserved"] = hashes_after == hashes_before
    (output_dir / "static_grouped_improvement_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result

