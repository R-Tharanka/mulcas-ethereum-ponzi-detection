"""Single, fixed sparse-SVM diagnostic on the duplicate-group-safe split."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from scripts.static_random_mulcas import best_f1_threshold, evaluate
from src.experiments.static_grouped_improvement import clone_cap
from src.experiments.static_grouped_mulcas import (
    DATASET,
    ROOT,
    RESULT_DIR as PARENT_DIR,
    grouped_split,
    load_grouped_data,
    sha256_file,
)


CONFIG = ROOT / "configs" / "experiments" / "static_grouped_sparse_svm.yaml"
OUTPUT_DIR = ROOT / "results" / "experiments" / "static_grouped_sparse_svm"
PARENT_RESULT = PARENT_DIR / "static_grouped_mulcas_results.json"


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run(config_path: Path = CONFIG, output_dir: Path = OUTPUT_DIR) -> dict:
    config = load_config(config_path)
    expected = {
        "dataset": config["dataset_sha256"],
        "parent": config["parent_result_sha256"],
    }
    observed_before = {"dataset": sha256_file(DATASET), "parent": sha256_file(PARENT_RESULT)}
    if observed_before != expected:
        raise RuntimeError(f"Frozen input mismatch: {observed_before}")

    started = time.perf_counter()
    frame = load_grouped_data(DATASET)
    parts = grouped_split(frame, int(config["seed"]))
    fit = clone_cap(parts["fit"])
    validation = parts["validation"]
    test = parts["test"]

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        min_df=int(config["features"]["min_df"]),
        max_features=int(config["features"]["max_features"]),
        ngram_range=tuple(config["features"]["ngram_range"]),
        dtype=np.float32,
    )
    x_fit = vectorizer.fit_transform(fit.opcode)
    model = LinearSVC(
        C=float(config["model"]["C"]),
        class_weight=config["model"]["class_weight"],
        random_state=int(config["seed"]),
    )
    model.fit(x_fit, fit.label.to_numpy())
    validation_scores = model.decision_function(vectorizer.transform(validation.opcode))
    threshold, validation_f1 = best_f1_threshold(validation.label.to_numpy(), validation_scores)
    test_scores = model.decision_function(vectorizer.transform(test.opcode))
    test_metrics = evaluate(test.label.to_numpy(), test_scores, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[["row_index", "address", "label", "duplicate_group", "opcode_sha256"]].copy()
    predictions["score"] = test_scores
    predictions["prediction"] = (test_scores >= threshold).astype(int)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    result = {
        "status": "complete",
        "experiment": "clone_capped_group_safe_sparse_linear_svm",
        "fit_rows": len(fit),
        "validation_f1": validation_f1,
        "validation_threshold": threshold,
        "test": test_metrics,
        "parent_group_safe_test": parent["test"],
        "test_minus_parent": {
            key: float(test_metrics[key] - parent["test"][key])
            for key in ("precision", "recall", "f1", "average_precision")
        },
        "guards": {
            "exact_group_safe_split": True,
            "p5_loaded": False,
            "track_b_loaded": False,
            "test_used_for_threshold": False,
            "hyperparameter_search": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    observed_after = {"dataset": sha256_file(DATASET), "parent": sha256_file(PARENT_RESULT)}
    result["frozen_inputs_preserved"] = observed_after == observed_before
    (output_dir / "static_grouped_sparse_svm_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result

