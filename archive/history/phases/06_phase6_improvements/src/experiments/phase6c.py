"""Controlled decision-threshold optimization for Phase 6C."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import (
    _git_commit,
    build_feature_pair,
    fold_metadata,
    load_development_data,
    sha256_file,
    verify_frozen_manifest,
)
from src.experiments.phase6b import baseline_parameters
from src.experiments.temporal_cv import make_positive_occurrence_folds
from src.models.baselines import build_baseline_model, positive_class_scores
from src.models.run_baselines import load_baseline_config
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/experiments/phase6c/threshold_optimization.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset", "split_membership", "paper_feature_config", "paper_model_config",
        "paper_mulcas_config", "frozen_phase55_manifest", "frozen_phase6a_manifest",
        "frozen_phase6b_manifest", "output_dir",
    ):
        config[key] = _resolve(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["selection"]["final_test_usage"] != "prohibited":
        raise ValueError("Phase 6C final-test usage must be prohibited.")
    return config


def candidate_thresholds(scores: np.ndarray, default: float, grid_size: int) -> np.ndarray:
    """Return a bounded deterministic quantile grid including the native default."""
    if grid_size < 3:
        raise ValueError("Threshold grid must contain at least three quantiles.")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a non-empty finite vector.")
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, grid_size))
    return np.unique(np.append(quantiles, float(default)))


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Evaluate one operating threshold while retaining continuous-score PR-AUC."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    return binary_classification_metrics(labels, predictions, scores)


def select_threshold(
    records: list[dict[str, Any]], *, minimum_precision: float, default: float
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select an eligible threshold, plus an always-available diagnostic fallback."""
    ranking = lambda r: (
        r["pooled_cv_f1"], r["pooled_cv_recall"], r["pooled_cv_balanced_accuracy"],
        -abs(r["threshold"] - default), -r["threshold"],
    )
    eligible = [record for record in records if record["pooled_cv_precision"] >= minimum_precision]
    return (max(eligible, key=ranking) if eligible else None), max(records, key=ranking)


def _score_model(model: Any, features: Any, score_type: str) -> np.ndarray:
    if score_type == "decision_function":
        return np.asarray(model.decision_function(features), dtype=float).reshape(-1)
    if score_type == "probability":
        return np.asarray(model.predict_proba(features), dtype=float)[:, 1]
    raise ValueError(f"Unknown score type: {score_type}")


def run_phase6c(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    seed = int(config["seed"])
    set_random_seed(seed)
    phase55 = verify_frozen_manifest(config["frozen_phase55_manifest"])
    phase6a = verify_frozen_manifest(config["frozen_phase6a_manifest"])
    phase6b = verify_frozen_manifest(config["frozen_phase6b_manifest"])
    partitions = load_development_data(config["dataset"], config["split_membership"])
    training, validation = partitions["train"], partitions["validation"]
    indexed = training.set_index("row_index", drop=False)
    folds = make_positive_occurrence_folds(
        training, config["temporal_cv"]["train_positive_ends"],
        config["temporal_cv"]["validation_positive_ends"],
    )
    fold_definitions = [fold_metadata(fold, indexed) for fold in folds]
    feature_config = yaml.safe_load(config["paper_feature_config"].read_text(encoding="utf-8"))
    min_df = int(feature_config["ngram"]["min_df"])
    frozen_models = load_baseline_config(config["paper_model_config"])["models"]
    parameters = {
        family: baseline_parameters(family, frozen_models[family])
        for family in ("ridge", "svm", "xgboost")
    }
    dataset_hash = sha256_file(config["dataset"])
    common = {
        "hypothesis": config["hypothesis"], "git_commit": _git_commit(),
        "random_seed": seed, "dataset_sha256": dataset_hash,
        "split_config_sha256": phase55["configs/paper_aligned_split.yaml"],
        "feature_config_sha256": phase55["configs/paper_aligned_features.yaml"],
        "model_config_sha256": phase55["configs/paper_aligned_baseline_models.yaml"],
        "experiment_config_sha256": config["config_sha256"],
    }
    feature_cache: dict[tuple[int, str], tuple[Any, Any, dict[str, Any]]] = {}
    threshold_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    pooled_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family, model_spec in config["models"].items():
        labels_parts, score_parts = [], []
        for fold in folds:
            train = indexed.loc[list(fold.train_indices)].reset_index(drop=True)
            later = indexed.loc[list(fold.validation_indices)].reset_index(drop=True)
            cache_key = (fold.fold, model_spec["feature"])
            if cache_key not in feature_cache:
                feature_cache[cache_key] = build_feature_pair(
                    model_spec["feature"], train, later, ngram_min_df=min_df)
            train_x, later_x, _ = feature_cache[cache_key]
            model = build_baseline_model(family, parameters[family], seed)
            model.fit(train_x, train["label"].to_numpy(int))
            labels_parts.append(later["label"].to_numpy(int))
            score_parts.append(_score_model(model, later_x, model_spec["score_type"]))
        pooled_labels = np.concatenate(labels_parts)
        pooled_scores = np.concatenate(score_parts)
        pooled_data[family] = (pooled_labels, pooled_scores)
        default = float(model_spec["default_threshold"])
        thresholds = candidate_thresholds(
            pooled_scores, default, int(config["selection"]["quantile_grid_size"])
        )
        family_records = []
        model_pr_auc = float(average_precision_score(pooled_labels, pooled_scores))
        for index, threshold in enumerate(thresholds):
            metrics = threshold_metrics(pooled_labels, pooled_scores, float(threshold))
            record = {
                "experiment_id": f"phase6c_{family}_threshold_{index:03d}", **common,
                "model_family": family, "feature_representation": model_spec["feature"],
                "score_type": model_spec["score_type"], "threshold": float(threshold),
                "is_model_default": bool(np.isclose(threshold, default, rtol=0, atol=1e-12)),
                "pooled_cv_precision": metrics["precision"], "pooled_cv_recall": metrics["recall"],
                "pooled_cv_f1": metrics["f1"], "pooled_cv_balanced_accuracy": metrics["balanced_accuracy"],
                "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"], "tn": metrics["tn"],
                "eligible": bool(metrics["precision"] >= float(config["selection"]["minimum_cv_precision"])),
            }
            threshold_records.append(record); family_records.append(record)
        selected, fallback = select_threshold(
            family_records, minimum_precision=float(config["selection"]["minimum_cv_precision"]), default=default)
        chosen = selected or fallback
        candidates.append({
            "model": family, "feature_representation": model_spec["feature"],
            "score_type": model_spec["score_type"], "native_default_threshold": default,
            "cv_selected_threshold": chosen["threshold"],
            "threshold_delta": chosen["threshold"] - default,
            "pooled_cv_precision": chosen["pooled_cv_precision"],
            "pooled_cv_recall": chosen["pooled_cv_recall"], "pooled_cv_f1": chosen["pooled_cv_f1"],
            "pooled_cv_balanced_accuracy": chosen["pooled_cv_balanced_accuracy"],
            "model_level_pooled_cv_pr_auc": model_pr_auc,
            "eligible_threshold_found": selected is not None,
            "selection_reason": (
                "maximum pooled temporal-CV F1 among precision >= 0.80 thresholds; "
                "ties: recall, balanced accuracy, closest to native default"
                if selected else
                "diagnostic fallback only: no threshold met pooled temporal-CV precision >= 0.80"
            ),
        })

    full_features = {
        feature: build_feature_pair(feature, training, validation, ngram_min_df=min_df)
        for feature in {spec["feature"] for spec in config["models"].values()}
    }
    validations = []
    for candidate in candidates:
        family = candidate["model"]
        train_x, validation_x, audit = full_features[candidate["feature_representation"]]
        model = build_baseline_model(family, parameters[family], seed)
        model.fit(train_x, training["label"].to_numpy(int))
        scores = _score_model(model, validation_x, candidate["score_type"])
        labels = validation["label"].to_numpy(int)
        default_metrics = threshold_metrics(labels, scores, candidate["native_default_threshold"])
        selected_metrics = threshold_metrics(labels, scores, candidate["cv_selected_threshold"])
        deltas = {
            key: selected_metrics[key] - default_metrics[key]
            for key in ("precision", "recall", "f1", "balanced_accuracy", "tp", "fp", "fn", "tn")
        }
        validations.append({
            "model": family, "feature_representation": candidate["feature_representation"],
            "score_type": candidate["score_type"],
            "native_default_threshold": candidate["native_default_threshold"],
            "cv_selected_threshold": candidate["cv_selected_threshold"],
            "eligible_threshold_found": candidate["eligible_threshold_found"],
            "default_threshold_metrics": default_metrics,
            "selected_threshold_metrics": selected_metrics, "metric_changes": deltas,
            "model_level_validation_pr_auc": float(average_precision_score(labels, scores)),
            "validation_rows": int(len(validation)), **audit,
        })
    reference = config["reference_validation"]
    eligible_validations = [v for v in validations if v["eligible_threshold_found"]]
    supported = any(
        v["selected_threshold_metrics"]["precision"] >= 0.80
        and (v["selected_threshold_metrics"]["f1"] > float(reference["f1"])
             or v["selected_threshold_metrics"]["recall"] > float(reference["recall"]))
        for v in eligible_validations
    )
    mulcas_hash = phase55["configs/paper_aligned_mulcas.yaml"]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(), "phase": "6C",
        "hypothesis": config["hypothesis"],
        "protocol": {
            "intervention": "classifier operating threshold only", "sampling": "none",
            "class_weighting": "none", "probability_calibration": "none",
            "threshold_selection": "pooled forward-only temporal out-of-fold scores",
            "validation_usage": "one-time evaluation after CV selection",
            "final_test_usage": "prohibited",
            "mulcas_config_sha256_before_and_after": mulcas_hash,
        },
        "provenance": {**common, "phase55_hashes": phase55, "phase6a_hashes": phase6a, "phase6b_hashes": phase6b},
        "temporal_cv_folds": fold_definitions, "threshold_search_records": threshold_records,
        "selected_candidates": candidates, "validation_records": validations,
        "reference_validation": reference, "h6c_supported": supported,
        "total_runtime_seconds": time.perf_counter() - started,
    }


def _flatten_validation(record: dict[str, Any]) -> dict[str, Any]:
    flat = {k: v for k, v in record.items() if k not in {"default_threshold_metrics", "selected_threshold_metrics", "metric_changes"}}
    for prefix, values in (("default", record["default_threshold_metrics"]), ("selected", record["selected_threshold_metrics"]), ("change", record["metric_changes"])):
        flat.update({f"{prefix}_{key}": value for key, value in values.items()})
    return flat


def save_results(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / filename for name, filename in {
        "json": "phase6c_results.json", "search": "phase6c_threshold_search.csv",
        "candidates": "phase6c_candidates.csv", "validation": "phase6c_validation.csv",
        "report": "phase6c_report.md", "folds": "temporal_cv_folds.json",
    }.items()}
    paths["json"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results["threshold_search_records"]).to_csv(paths["search"], index=False)
    pd.DataFrame(results["selected_candidates"]).to_csv(paths["candidates"], index=False)
    pd.DataFrame([_flatten_validation(v) for v in results["validation_records"]]).to_csv(paths["validation"], index=False)
    paths["folds"].write_text(json.dumps(results["temporal_cv_folds"], indent=2), encoding="utf-8")
    rows = ["| Model | Default | Selected | Eligible | CV precision | CV recall | CV F1 | Default val F1 | Selected val F1 |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    validations = {v["model"]: v for v in results["validation_records"]}
    for candidate in results["selected_candidates"]:
        validation = validations[candidate["model"]]
        rows.append(f"| {candidate['model']} | {candidate['native_default_threshold']:.6g} | {candidate['cv_selected_threshold']:.6g} | {candidate['eligible_threshold_found']} | {candidate['pooled_cv_precision']:.6f} | {candidate['pooled_cv_recall']:.6f} | {candidate['pooled_cv_f1']:.6f} | {validation['default_threshold_metrics']['f1']:.6f} | {validation['selected_threshold_metrics']['f1']:.6f} |")
    paths["report"].write_text(
        "# Phase 6C: Controlled Decision-Threshold Optimization\n\nOnly native classifier operating thresholds were changed. MulCas cascade theta, lambda, viewer order, routing, and voting were not changed. Final-test rows and labels were not loaded.\n\n## Results\n\n"
        + "\n".join(rows)
        + f"\n\n## Outcome\n\n- H6C supported: `{str(results['h6c_supported']).lower()}`\n- Eligibility: pooled temporal-CV precision >= 0.80.\n- PR-AUC is computed from continuous scores and is invariant across thresholds for each fitted score vector.\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6C threshold optimization.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    for name, path in save_results(run_phase6c(config), config["output_dir"]).items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
