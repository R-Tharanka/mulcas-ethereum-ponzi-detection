"""Controlled cost-sensitive learning experiments for Phase 6A."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.temporal_cv import TemporalFold, make_positive_occurrence_folds
from src.features.developer import ChronologicalCreatorTransformer
from src.features.opcode import (
    make_opcode_term_count_vectorizer,
    make_paper_aligned_ngram_tfidf_vectorizer,
)
from src.models.baselines import build_baseline_model, positive_class_scores
from src.models.run_baselines import load_baseline_config
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "experiments" / "phase6a" / "cost_sensitive.yaml"
)
SPLIT_NAMES = ("train", "validation")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset",
        "split_membership",
        "paper_feature_config",
        "paper_model_config",
        "frozen_manifest",
        "output_dir",
    ):
        config[key] = _resolve(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["selection"]["final_test_usage"] != "prohibited":
        raise ValueError("Phase 6A final-test usage must be prohibited.")
    return config


def verify_frozen_manifest(path: Path) -> dict[str, str]:
    """Fail before experimentation if any frozen Phase 5.5 file changed."""
    expected = yaml.safe_load(path.read_text(encoding="utf-8"))["sha256"]
    observed: dict[str, str] = {}
    for relative_path, expected_hash in expected.items():
        target = REPO_ROOT / relative_path
        if not target.is_file():
            raise FileNotFoundError(f"Frozen file is missing: {relative_path}")
        observed_hash = sha256_file(target)
        if observed_hash != expected_hash:
            raise RuntimeError(f"Frozen Phase 5.5 file changed: {relative_path}")
        observed[relative_path] = observed_hash
    return observed


def load_development_data(dataset: Path, membership_path: Path) -> dict[str, pd.DataFrame]:
    """Load train and validation only; final-test rows are excluded immediately."""
    membership = pd.read_csv(
        membership_path,
        usecols=["row_index", "split"],
    )
    development = membership[membership["split"].isin(SPLIT_NAMES)].copy()
    source = pd.read_csv(
        dataset,
        usecols=["address", "opcode", "creator", "createdBlockNumber", "label"],
    )
    source["row_index"] = source.index.astype(int)
    frame = development.merge(source, on="row_index", how="left", validate="one_to_one")
    if frame.isna().any().any():
        raise ValueError("Phase 6A development data contains missing values.")
    frame = frame.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
    partitions = {
        name: frame[frame["split"] == name].copy().reset_index(drop=True)
        for name in SPLIT_NAMES
    }
    if any(partitions[name].empty for name in SPLIT_NAMES):
        raise ValueError("Phase 6A requires non-empty train and validation partitions.")
    return partitions


def fold_metadata(fold: TemporalFold, indexed_training: pd.DataFrame) -> dict[str, Any]:
    train = indexed_training.loc[list(fold.train_indices)]
    validation = indexed_training.loc[list(fold.validation_indices)]
    return {
        "fold": fold.fold,
        "train_rows": int(len(train)),
        "train_positive": int(train["label"].sum()),
        "train_negative": int(len(train) - train["label"].sum()),
        "train_block_min": int(train["createdBlockNumber"].min()),
        "train_block_max": int(train["createdBlockNumber"].max()),
        "validation_rows": int(len(validation)),
        "validation_positive": int(validation["label"].sum()),
        "validation_negative": int(len(validation) - validation["label"].sum()),
        "validation_block_min": int(validation["createdBlockNumber"].min()),
        "validation_block_max": int(validation["createdBlockNumber"].max()),
        "chronology_assertion": bool(
            train["createdBlockNumber"].max()
            < validation["createdBlockNumber"].min()
        ),
    }


def build_feature_pair(
    feature_name: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    ngram_min_df: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Fit static and creator transformers on the supplied past-only rows."""
    train_documents = train["opcode"].astype(str).tolist()
    evaluation_documents = evaluation["opcode"].astype(str).tolist()
    if feature_name == "opcode_term_count":
        vectorizer = make_opcode_term_count_vectorizer()
    elif feature_name == "opcode_ngram_tfidf_1_3_df_filtered":
        vectorizer = make_paper_aligned_ngram_tfidf_vectorizer(ngram_min_df)
    else:
        raise ValueError(f"Unsupported Phase 6A feature: {feature_name}")
    train_static = vectorizer.fit_transform(train_documents).tocsr()
    evaluation_static = vectorizer.transform(evaluation_documents).tocsr()

    creator = ChronologicalCreatorTransformer()
    train_creator = creator.fit_transform(train)
    evaluation_creator = creator.transform(
        evaluation.loc[:, ["creator", "createdBlockNumber"]]
    )
    return (
        hstack([train_static, train_creator], format="csr"),
        hstack([evaluation_static, evaluation_creator], format="csr"),
        {
            "feature_fit_rows": int(len(train)),
            "feature_fit_block_max": int(train["createdBlockNumber"].max()),
            "evaluation_block_min": int(evaluation["createdBlockNumber"].min()),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "creator_positive_state_size": int(len(creator.positive_creators_)),
        },
    )


def candidate_parameters(
    model: str,
    base_parameters: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], Any, float | None]:
    """Apply only the configured class-cost parameter to frozen model settings."""
    parameters = dict(base_parameters)
    class_weight: Any = None
    scale_pos_weight: float | None = None
    if model in {"ridge", "svm"}:
        if "positive_weight" in candidate:
            class_weight = {0: 1.0, 1: float(candidate["positive_weight"])}
        else:
            class_weight = candidate.get("class_weight")
        parameters["class_weight"] = class_weight
    elif model == "xgboost":
        scale_pos_weight = float(candidate["scale_pos_weight"])
        parameters["scale_pos_weight"] = scale_pos_weight
    else:
        raise ValueError(f"Unsupported Phase 6A model: {model}")
    return parameters, class_weight, scale_pos_weight


def evaluate_once(
    model_family: str,
    parameters: dict[str, Any],
    seed: int,
    train_features: Any,
    train_labels: np.ndarray,
    evaluation_features: Any,
    evaluation_labels: np.ndarray,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    model = build_baseline_model(model_family, parameters, seed)
    model.fit(train_features, train_labels)
    predictions = np.asarray(model.predict(evaluation_features), dtype=int)
    scores = positive_class_scores(model, evaluation_features)
    elapsed = time.perf_counter() - started
    return binary_classification_metrics(evaluation_labels, predictions, scores), elapsed


def aggregate_cv(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for metric in ("precision", "recall", "f1", "pr_auc"):
        values = np.asarray([fold[metric] for fold in fold_metrics], dtype=float)
        aggregate[f"cv_{metric}_mean"] = float(values.mean())
        aggregate[f"cv_{metric}_std"] = float(values.std(ddof=0))
    return aggregate


def select_candidate(
    records: list[dict[str, Any]],
    *,
    minimum_precision: float,
) -> tuple[dict[str, Any], bool]:
    """Select by CV F1 with predeclared precision constraint and tie breakers."""
    eligible = [r for r in records if r["cv_precision_mean"] >= minimum_precision]
    precision_fallback = not eligible
    pool = eligible or records
    selected = max(
        pool,
        key=lambda r: (
            r["cv_f1_mean"],
            r["cv_recall_mean"],
            r["cv_pr_auc_mean"],
            r["cv_precision_mean"],
            r["experiment_id"],
        ),
    )
    return selected, precision_fallback


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_phase6a(config: dict[str, Any]) -> dict[str, Any]:
    """Run CV selection and one validation evaluation per model family."""
    run_started = time.perf_counter()
    seed = int(config["seed"])
    set_random_seed(seed)
    frozen_hashes = verify_frozen_manifest(config["frozen_manifest"])
    partitions = load_development_data(config["dataset"], config["split_membership"])
    training = partitions["train"]
    validation = partitions["validation"]
    indexed_training = training.set_index("row_index", drop=False)
    cv_config = config["temporal_cv"]
    folds = make_positive_occurrence_folds(
        training,
        cv_config["train_positive_ends"],
        cv_config["validation_positive_ends"],
    )
    fold_definitions = [fold_metadata(fold, indexed_training) for fold in folds]

    feature_config = yaml.safe_load(
        config["paper_feature_config"].read_text(encoding="utf-8")
    )
    ngram_min_df = int(feature_config["ngram"]["min_df"])
    baseline_config = load_baseline_config(config["paper_model_config"])
    model_parameters = baseline_config["models"]
    feature_cache: dict[tuple[int, str], tuple[Any, Any, dict[str, Any]]] = {}
    for fold in folds:
        fold_train = indexed_training.loc[list(fold.train_indices)].reset_index(drop=True)
        fold_validation = indexed_training.loc[
            list(fold.validation_indices)
        ].reset_index(drop=True)
        for feature_name in set(config["features"].values()):
            feature_cache[(fold.fold, feature_name)] = build_feature_pair(
                feature_name,
                fold_train,
                fold_validation,
                ngram_min_df=ngram_min_df,
            )

    dataset_hash = sha256_file(config["dataset"])
    common = {
        "hypothesis": config["hypothesis"],
        "git_commit": _git_commit(),
        "random_seed": seed,
        "dataset_sha256": dataset_hash,
        "split_config_sha256": frozen_hashes["configs/paper_aligned_split.yaml"],
        "feature_config_sha256": frozen_hashes["configs/paper_aligned_features.yaml"],
        "experiment_config_sha256": config["config_sha256"],
        "decision_threshold": "estimator_default_unchanged",
        "final_test_used": False,
    }
    records: list[dict[str, Any]] = []
    selected_by_family: dict[str, str] = {}
    selection_fallbacks: dict[str, bool] = {}
    for model_family in ("ridge", "svm", "xgboost"):
        feature_name = config["features"][model_family]
        family_records: list[dict[str, Any]] = []
        for candidate in config["cost_candidates"][model_family]:
            parameters, class_weight, scale_pos_weight = candidate_parameters(
                model_family, model_parameters[model_family], candidate
            )
            fold_results = []
            runtime_seconds = 0.0
            for fold in folds:
                train_matrix, validation_matrix, feature_audit = feature_cache[
                    (fold.fold, feature_name)
                ]
                fold_train = indexed_training.loc[list(fold.train_indices)]
                fold_validation = indexed_training.loc[list(fold.validation_indices)]
                metrics, elapsed = evaluate_once(
                    model_family,
                    parameters,
                    seed,
                    train_matrix,
                    fold_train["label"].to_numpy(dtype=int),
                    validation_matrix,
                    fold_validation["label"].to_numpy(dtype=int),
                )
                runtime_seconds += elapsed
                fold_results.append(
                    {
                        "fold": fold.fold,
                        **feature_audit,
                        **metrics,
                        "runtime_seconds": elapsed,
                    }
                )
            record = {
                "experiment_id": f"phase6a_{model_family}_{candidate['name']}",
                **common,
                "model": model_family,
                "feature_representation": feature_name,
                "class_weight": class_weight,
                "scale_pos_weight": scale_pos_weight,
                "parameters": parameters,
                "temporal_cv_fold_metrics": fold_results,
                **aggregate_cv(fold_results),
                "selected_by_temporal_cv": False,
                "validation_precision": None,
                "validation_recall": None,
                "validation_f1": None,
                "validation_pr_auc": None,
                "validation_balanced_accuracy": None,
                "validation_tn": None,
                "validation_fp": None,
                "validation_fn": None,
                "validation_tp": None,
                "runtime_seconds": runtime_seconds,
            }
            records.append(record)
            family_records.append(record)
        selected, fallback = select_candidate(
            family_records,
            minimum_precision=float(config["selection"]["minimum_cv_precision"]),
        )
        selected["selected_by_temporal_cv"] = True
        selected_by_family[model_family] = selected["experiment_id"]
        selection_fallbacks[model_family] = fallback

    full_feature_cache: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for feature_name in set(config["features"].values()):
        full_feature_cache[feature_name] = build_feature_pair(
            feature_name,
            training,
            validation,
            ngram_min_df=ngram_min_df,
        )
    for record in records:
        if not record["selected_by_temporal_cv"]:
            continue
        train_matrix, validation_matrix, _ = full_feature_cache[
            record["feature_representation"]
        ]
        metrics, elapsed = evaluate_once(
            record["model"],
            record["parameters"],
            seed,
            train_matrix,
            training["label"].to_numpy(dtype=int),
            validation_matrix,
            validation["label"].to_numpy(dtype=int),
        )
        record.update(
            {
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_f1": metrics["f1"],
                "validation_pr_auc": metrics["pr_auc"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_tn": metrics["tn"],
                "validation_fp": metrics["fp"],
                "validation_fn": metrics["fn"],
                "validation_tp": metrics["tp"],
                "runtime_seconds": record["runtime_seconds"] + elapsed,
            }
        )

    selected_records = [r for r in records if r["selected_by_temporal_cv"]]
    minimum_validation_precision = float(
        config["selection"]["minimum_validation_precision"]
    )
    eligible = [
        r for r in selected_records if r["validation_precision"] >= minimum_validation_precision
    ]
    validation_fallback = not eligible
    best = max(
        eligible or selected_records,
        key=lambda r: (
            r["validation_f1"],
            r["validation_recall"],
            r["validation_pr_auc"],
            r["validation_precision"],
            r["experiment_id"],
        ),
    )
    reference = config["reference_validation"]
    h6a_supported = bool(
        best["validation_precision"] >= minimum_validation_precision
        and best["validation_recall"] > float(reference["recall"])
        and best["validation_f1"] > float(reference["f1"])
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": config["phase"],
        "experiment_name": config["experiment_name"],
        "hypothesis": config["hypothesis"],
        "protocol": {
            "parameter_selection": "forward_only_temporal_training_cv",
            "family_variant_evaluation": "phase_5_5_validation_once",
            "overall_candidate_selection": (
                "maximum validation F1 subject to precision >= "
                f"{minimum_validation_precision:.2f}; ties recall, PR-AUC, precision"
            ),
            "final_test_usage": "prohibited",
            "sampling": "none",
            "decision_threshold": "unchanged estimator default",
            "frozen_manifest_verified": True,
        },
        "provenance": {
            **common,
            "frozen_file_hashes": frozen_hashes,
        },
        "temporal_cv_folds": fold_definitions,
        "selected_by_family": selected_by_family,
        "cv_precision_constraint_fallback": selection_fallbacks,
        "validation_precision_constraint_fallback": validation_fallback,
        "reference_validation": reference,
        "best_phase6a_candidate": best["experiment_id"],
        "h6a_supported": h6a_supported,
        "records": records,
        "total_runtime_seconds": time.perf_counter() - run_started,
    }


def save_results(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase6a_results.json"
    candidates_path = output_dir / "phase6a_candidates.csv"
    validation_path = output_dir / "phase6a_validation.csv"
    folds_path = output_dir / "temporal_cv_folds.json"
    report_path = output_dir / "phase6a_report.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    flat_records = []
    for record in results["records"]:
        flat = {
            key: value
            for key, value in record.items()
            if key not in {"temporal_cv_fold_metrics", "parameters"}
        }
        flat["class_weight"] = json.dumps(record["class_weight"], sort_keys=True)
        flat["parameters"] = json.dumps(record["parameters"], sort_keys=True)
        flat_records.append(flat)
    candidates = pd.DataFrame(flat_records)
    candidates.to_csv(candidates_path, index=False)
    candidates[candidates["selected_by_temporal_cv"]].to_csv(
        validation_path, index=False
    )
    folds_path.write_text(
        json.dumps(results["temporal_cv_folds"], indent=2), encoding="utf-8"
    )
    selected = [r for r in results["records"] if r["selected_by_temporal_cv"]]
    rows = [
        "| Model | Cost | CV precision | CV recall | CV F1 | CV PR-AUC | "
        "Validation precision | Validation recall | Validation F1 | Validation PR-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in selected:
        cost = record["class_weight"] or record["scale_pos_weight"]
        rows.append(
            f"| {record['model']} | `{cost}` | {record['cv_precision_mean']:.6f} | "
            f"{record['cv_recall_mean']:.6f} | {record['cv_f1_mean']:.6f} | "
            f"{record['cv_pr_auc_mean']:.6f} | {record['validation_precision']:.6f} | "
            f"{record['validation_recall']:.6f} | {record['validation_f1']:.6f} | "
            f"{record['validation_pr_auc']:.6f} |"
        )
    report = f"""# Phase 6A: Controlled Cost-Sensitive Learning

Final-test rows and metrics were not used or generated. Parameters were selected
with forward-only temporal CV, followed by one validation evaluation per family.

## Selected Variants

{chr(10).join(rows)}

## Outcome

- Best candidate: `{results['best_phase6a_candidate']}`
- H6A supported: `{str(results['h6a_supported']).lower()}`
- Selection criterion: {results['protocol']['overall_candidate_selection']}
- CV precision-constraint fallback by family: `{json.dumps(results['cv_precision_constraint_fallback'], sort_keys=True)}`
- Validation precision-constraint fallback: `{str(results['validation_precision_constraint_fallback']).lower()}`

H6A is assessed against the frozen Phase 5.5 **validation** metrics because
Phase 6 candidates may not be evaluated against the final-test partition.
No candidate satisfying the configured precision floor was available when a
fallback is reported; the named candidate is then a diagnostic ranking, not a
claim that the precision-constrained objective was achieved.
"""
    report_path.write_text(report, encoding="utf-8")
    return {
        "json": json_path,
        "candidates_csv": candidates_path,
        "validation_csv": validation_path,
        "folds_json": folds_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6A cost experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = save_results(run_phase6a(config), config["output_dir"])
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
