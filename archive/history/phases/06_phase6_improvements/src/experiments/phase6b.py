"""Controlled training-data sampling experiments for Phase 6B."""

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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.sparse import issparse

from src.experiments.phase6a import (
    _git_commit,
    aggregate_cv,
    build_feature_pair,
    evaluate_once,
    fold_metadata,
    load_development_data,
    select_candidate,
    sha256_file,
    verify_frozen_manifest,
)
from src.experiments.temporal_cv import make_positive_occurrence_folds
from src.models.run_baselines import load_baseline_config
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/experiments/phase6b/training_sampling.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset", "split_membership", "paper_feature_config", "paper_model_config",
                "frozen_phase55_manifest", "frozen_phase6a_manifest", "output_dir"):
        config[key] = _resolve(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["selection"]["final_test_usage"] != "prohibited":
        raise ValueError("Phase 6B final-test usage must be prohibited.")
    return config


def baseline_parameters(model: str, configured: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(configured)
    if model in {"ridge", "svm"}:
        if parameters.get("class_weight") is not None:
            raise ValueError("Phase 6B classifiers must be unweighted.")
    elif model == "xgboost":
        if float(parameters.get("scale_pos_weight", 1.0)) != 1.0:
            raise ValueError("Phase 6B scale_pos_weight must remain 1.")
        parameters["scale_pos_weight"] = 1.0
    return parameters


def apply_training_sampler(
    features: Any,
    labels: np.ndarray,
    *,
    sampler_type: str,
    sampling_strategy: float | None,
    requested_ratio: str,
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Resample a training matrix only and return a complete class-count audit."""
    labels = np.asarray(labels, dtype=int)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    sparse_input = bool(issparse(features))
    status = "applied"
    if sampler_type == "none":
        sampled_features, sampled_labels = features, labels.copy()
        status = "natural_no_sampling"
    else:
        if sampling_strategy is None:
            raise ValueError("A numeric sampling strategy is required.")
        current_ratio = positive / negative
        if float(sampling_strategy) < current_ratio - 1e-12:
            sampled_features, sampled_labels = features, labels.copy()
            status = "unreachable_from_current_ratio"
        else:
            sampler_class = RandomUnderSampler if sampler_type == "random_under" else RandomOverSampler
            if sampler_type not in {"random_under", "random_over"}:
                raise ValueError(f"Unknown sampler: {sampler_type}")
            sampler = sampler_class(sampling_strategy=float(sampling_strategy), random_state=seed)
            sampled_features, sampled_labels = sampler.fit_resample(features, labels)
    post_positive = int(np.asarray(sampled_labels).sum())
    post_negative = int(len(sampled_labels) - post_positive)
    if sparse_input and not issparse(sampled_features):
        raise AssertionError("Sampling unexpectedly densified a sparse matrix.")
    return sampled_features, np.asarray(sampled_labels, dtype=int), {
        "sampler_type": sampler_type,
        "requested_positive_negative_ratio": requested_ratio,
        "sampler_api_sampling_strategy": sampling_strategy,
        "sampling_status": status,
        "original_training_positive": positive,
        "original_training_negative": negative,
        "post_sampling_positive": post_positive,
        "post_sampling_negative": post_negative,
        "rows_removed": max(0, len(labels) - len(sampled_labels)),
        "rows_duplicated": max(0, len(sampled_labels) - len(labels)),
        "sparse_input_preserved": sparse_input and bool(issparse(sampled_features)),
    }


def run_phase6b(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    seed = int(config["seed"])
    set_random_seed(seed)
    phase55_hashes = verify_frozen_manifest(config["frozen_phase55_manifest"])
    phase6a_hashes = verify_frozen_manifest(config["frozen_phase6a_manifest"])
    partitions = load_development_data(config["dataset"], config["split_membership"])
    training, validation = partitions["train"], partitions["validation"]
    indexed = training.set_index("row_index", drop=False)
    folds = make_positive_occurrence_folds(
        training, config["temporal_cv"]["train_positive_ends"],
        config["temporal_cv"]["validation_positive_ends"])
    fold_definitions = [fold_metadata(fold, indexed) for fold in folds]
    feature_cfg = yaml.safe_load(config["paper_feature_config"].read_text(encoding="utf-8"))
    min_df = int(feature_cfg["ngram"]["min_df"])
    model_cfg = load_baseline_config(config["paper_model_config"])["models"]
    params = {name: baseline_parameters(name, model_cfg[name]) for name in ("ridge", "svm", "xgboost")}
    feature_cache: dict[tuple[int, str], tuple[Any, Any, dict[str, Any]]] = {}
    for fold in folds:
        past = indexed.loc[list(fold.train_indices)].reset_index(drop=True)
        later = indexed.loc[list(fold.validation_indices)].reset_index(drop=True)
        for feature in set(config["features"].values()):
            feature_cache[(fold.fold, feature)] = build_feature_pair(feature, past, later, ngram_min_df=min_df)

    dataset_hash = sha256_file(config["dataset"])
    git_commit = _git_commit()
    split_hash = phase55_hashes["configs/paper_aligned_split.yaml"]
    records: list[dict[str, Any]] = []
    selected_by_family: dict[str, str] = {}
    fallbacks: dict[str, bool] = {}
    for family in ("ridge", "svm", "xgboost"):
        family_records = []
        feature = config["features"][family]
        for candidate in config["sampling_candidates"]:
            fold_runs, elapsed_total = [], 0.0
            for fold in folds:
                train_x, cv_x, feature_audit = feature_cache[(fold.fold, feature)]
                fold_train = indexed.loc[list(fold.train_indices)]
                fold_cv = indexed.loc[list(fold.validation_indices)]
                sampled_x, sampled_y, sampling = apply_training_sampler(
                    train_x, fold_train["label"].to_numpy(int),
                    sampler_type=candidate["sampler"],
                    sampling_strategy=candidate["sampling_strategy"],
                    requested_ratio=candidate["requested_ratio"], seed=seed)
                metrics, elapsed = evaluate_once(
                    family, params[family], seed, sampled_x, sampled_y, cv_x,
                    fold_cv["label"].to_numpy(int))
                elapsed_total += elapsed
                meta = fold_definitions[fold.fold - 1]
                fold_runs.append({
                    "experiment_id": f"phase6b_{family}_{candidate['name']}",
                    "hypothesis": config["hypothesis"], "model_family": family,
                    "feature_representation": feature, **sampling, "fold_id": fold.fold,
                    "train_block_range": [meta["train_block_min"], meta["train_block_max"]],
                    "cv_block_range": [meta["validation_block_min"], meta["validation_block_max"]],
                    "random_seed": seed, "dataset_sha256": dataset_hash,
                    "git_commit": git_commit,
                    "split_config_sha256": split_hash, "experiment_config_sha256": config["config_sha256"],
                    "class_weight": None, "scale_pos_weight": 1.0 if family == "xgboost" else None,
                    "cv_rows_before_sampling": len(fold_cv), "cv_rows_after_sampling": len(fold_cv),
                    **feature_audit, **metrics, "runtime_seconds": elapsed})
            record = {
                "experiment_id": f"phase6b_{family}_{candidate['name']}",
                "model_family": family, "feature_representation": feature,
                "sampler_type": candidate["sampler"],
                "requested_positive_negative_ratio": candidate["requested_ratio"],
                "sampler_api_sampling_strategy": candidate["sampling_strategy"],
                "temporal_cv_fold_metrics": fold_runs, **aggregate_cv(fold_runs),
                "selected_by_temporal_cv": False, "selection_reason": None,
                "runtime_seconds": elapsed_total,
            }
            records.append(record); family_records.append(record)
        selected, fallback = select_candidate(family_records, minimum_precision=float(config["selection"]["minimum_cv_precision"]))
        selected["selected_by_temporal_cv"] = True
        selected["selection_reason"] = "maximum temporal-CV mean F1 subject to precision >= 0.80; tie-breakers recall, PR-AUC, precision"
        for record in family_records:
            if not record["selected_by_temporal_cv"]:
                record["selection_reason"] = "not selected by the predefined temporal-CV criterion"
        selected_by_family[family] = selected["experiment_id"]; fallbacks[family] = fallback

    full_cache = {feature: build_feature_pair(feature, training, validation, ngram_min_df=min_df)
                  for feature in set(config["features"].values())}
    validations = []
    for record in records:
        if not record["selected_by_temporal_cv"]: continue
        candidate = next(c for c in config["sampling_candidates"] if c["requested_ratio"] == record["requested_positive_negative_ratio"] and c["sampler"] == record["sampler_type"])
        train_x, validation_x, audit = full_cache[record["feature_representation"]]
        sampled_x, sampled_y, sampling = apply_training_sampler(
            train_x, training["label"].to_numpy(int), sampler_type=candidate["sampler"],
            sampling_strategy=candidate["sampling_strategy"], requested_ratio=candidate["requested_ratio"], seed=seed)
        metrics, elapsed = evaluate_once(record["model_family"], params[record["model_family"]], seed,
                                         sampled_x, sampled_y, validation_x, validation["label"].to_numpy(int))
        validations.append({"experiment_id": record["experiment_id"], "model_family": record["model_family"],
            "feature_representation": record["feature_representation"], **sampling, **audit,
            **metrics, "validation_rows_before_sampling": len(validation),
            "validation_rows_after_sampling": len(validation), "runtime_seconds": elapsed})
    eligible = [v for v in validations if v["precision"] >= float(config["selection"]["minimum_validation_precision"])]
    best = max(eligible or validations, key=lambda v: (v["f1"], v["recall"], v["pr_auc"], v["precision"], v["experiment_id"]))
    reference = config["reference_validation"]
    supported = bool(best["precision"] >= .80 and best["recall"] > float(reference["recall"]) and best["f1"] > float(reference["f1"]))
    return {"generated_at_utc": datetime.now(UTC).isoformat(), "phase": "6B", "hypothesis": config["hypothesis"],
        "protocol": {"preprocessing": "fold-local fit; later interval transform-only", "sampling": "training matrix only after developer feature construction",
                     "final_test_usage": "prohibited", "decision_threshold": "unchanged estimator default",
                     "unreachable_ratio_policy": "deterministic no-op with explicit status"},
        "provenance": {"random_seed": seed, "dataset_sha256": dataset_hash, "split_config_sha256": split_hash,
                       "git_commit": git_commit,
                       "feature_config_sha256": phase55_hashes["configs/paper_aligned_features.yaml"],
                       "experiment_config_sha256": config["config_sha256"], "phase55_hashes": phase55_hashes, "phase6a_hashes": phase6a_hashes},
        "temporal_cv_folds": fold_definitions, "selected_by_family": selected_by_family,
        "cv_precision_constraint_fallback": fallbacks, "reference_validation": reference,
        "best_phase6b_candidate": best["experiment_id"], "h6b_supported": supported,
        "records": records, "validation_records": validations, "total_runtime_seconds": time.perf_counter() - started}


def save_results(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / filename for name, filename in {
        "json": "phase6b_results.json", "cv": "phase6b_cv_runs.csv", "candidates": "phase6b_candidates.csv",
        "validation": "phase6b_validation.csv", "folds": "temporal_cv_folds.json", "verification": "preprocessing_verification.md", "report": "phase6b_report.md"}.items()}
    paths["json"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    cv_runs = [run for record in results["records"] for run in record["temporal_cv_fold_metrics"]]
    pd.DataFrame(cv_runs).to_csv(paths["cv"], index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != "temporal_cv_fold_metrics"} for r in results["records"]]).to_csv(paths["candidates"], index=False)
    pd.DataFrame(results["validation_records"]).to_csv(paths["validation"], index=False)
    paths["folds"].write_text(json.dumps(results["temporal_cv_folds"], indent=2), encoding="utf-8")
    paths["verification"].write_text("# Fold-local preprocessing verification\n\n`build_feature_pair` fits each vectorizer and chronological creator transformer only on the fold-training rows. The later interval calls `transform` only. Word2Vec/IDF transformer tests likewise verify that transform does not alter fitted vocabulary or IDF state. Sampling is then applied only to the completed training matrix, so removed or duplicated rows cannot change creator-history semantics.\n", encoding="utf-8")
    rows = ["| Model | Sampling | CV precision | CV recall | CV F1 | CV PR-AUC | Val precision | Val recall | Val F1 | Val PR-AUC |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    by_id = {v["experiment_id"]: v for v in results["validation_records"]}
    for r in results["records"]:
        if r["selected_by_temporal_cv"]:
            v = by_id[r["experiment_id"]]
            rows.append(f"| {r['model_family']} | {r['sampler_type']} {r['requested_positive_negative_ratio']} | {r['cv_precision_mean']:.6f} | {r['cv_recall_mean']:.6f} | {r['cv_f1_mean']:.6f} | {r['cv_pr_auc_mean']:.6f} | {v['precision']:.6f} | {v['recall']:.6f} | {v['f1']:.6f} | {v['pr_auc']:.6f} |")
    paths["report"].write_text("# Phase 6B: Controlled Training-Data Sampling\n\nFinal-test rows and metrics were not used. Ratios were selected using forward-only temporal CV; each selected family variant was evaluated once on validation.\n\n## Selected variants\n\n" + "\n".join(rows) + f"\n\n## Outcome\n\n- Best candidate: `{results['best_phase6b_candidate']}`\n- H6B supported: `{str(results['h6b_supported']).lower()}`\n- Frozen validation reference: precision {results['reference_validation']['precision']:.6f}, recall {results['reference_validation']['recall']:.6f}, F1 {results['reference_validation']['f1']:.6f}, PR-AUC {results['reference_validation']['pr_auc']:.6f}.\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6B sampling experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(); config = load_config(args.config)
    for name, path in save_results(run_phase6b(config), config["output_dir"]).items(): print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
