"""Controlled structural static-feature experiments for Phase 6D1."""

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
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import _git_commit, build_feature_pair, fold_metadata, load_development_data, sha256_file, verify_frozen_manifest
from src.experiments.phase6b import baseline_parameters
from src.experiments.temporal_cv import make_positive_occurrence_folds
from src.features.developer import ChronologicalCreatorTransformer
from src.features.structural import FEATURE_GROUPS, FEATURE_NAMES, GROUPS, feature_names_for_groups, transform_structural_features
from src.models.baselines import build_baseline_model, positive_class_scores
from src.models.run_baselines import load_baseline_config
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/experiments/phase6d1/structural_features.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset", "split_membership", "paper_feature_config", "paper_model_config", "paper_mulcas_config", "frozen_phase55_manifest", "frozen_phase6a_manifest", "frozen_phase6b_manifest", "frozen_phase6c_manifest", "output_dir"):
        config[key] = _resolve(config[key])
    config["config_sha256"] = sha256_file(path)
    if config["selection"]["final_test_usage"] != "prohibited":
        raise ValueError("Phase 6D1 final-test usage must be prohibited.")
    return config


def _creator_pair(train: pd.DataFrame, later: pd.DataFrame) -> tuple[csr_matrix, csr_matrix]:
    creator = ChronologicalCreatorTransformer()
    return csr_matrix(creator.fit_transform(train)), csr_matrix(creator.transform(later[["creator", "createdBlockNumber"]]))


def build_representation_pair(
    specification: dict[str, Any], family: str, train: pd.DataFrame, later: pd.DataFrame,
    baseline_pair: tuple[Any, Any], structural_pair: tuple[csr_matrix, csr_matrix],
    creator_pair: tuple[csr_matrix, csr_matrix],
) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
    """Build one sparse pair, fitting any scaler on historical training rows only."""
    groups = tuple(specification["groups"])
    indices = [sum(len(FEATURE_GROUPS[g]) for g in GROUPS[:GROUPS.index(group)]) + offset for group in groups for offset in range(len(FEATURE_GROUPS[group]))]
    train_struct = structural_pair[0][:, indices] if indices else csr_matrix((len(train), 0))
    later_struct = structural_pair[1][:, indices] if indices else csr_matrix((len(later), 0))
    scaler_type = None
    if groups and family in {"ridge", "svm"}:
        scaler = StandardScaler(with_mean=False)
        train_struct = scaler.fit_transform(train_struct).tocsr()
        later_struct = scaler.transform(later_struct).tocsr()
        scaler_type = "StandardScaler(with_mean=False), fold-training fit only"
    if specification["baseline"]:
        train_x, later_x = baseline_pair
        train_x = train_x if not groups else hstack([train_x, train_struct], format="csr")
        later_x = later_x if not groups else hstack([later_x, later_struct], format="csr")
        baseline_dimension = int(baseline_pair[0].shape[1])
    else:
        train_x = hstack([train_struct, creator_pair[0]], format="csr")
        later_x = hstack([later_struct, creator_pair[1]], format="csr")
        baseline_dimension = 0
    if not issparse(train_x) or not issparse(later_x):
        raise AssertionError("Phase 6D1 representations must remain sparse.")
    return train_x.tocsr(), later_x.tocsr(), {
        "included_feature_groups": list(groups),
        "ordered_structural_feature_names": list(feature_names_for_groups(groups)),
        "structural_dimension": len(indices), "baseline_dimension": baseline_dimension,
        "combined_dimension": int(train_x.shape[1]), "representation_type": "scipy.sparse.csr_matrix",
        "scaler_type": scaler_type, "scaler_fit_rows": int(len(train)) if scaler_type else 0,
        "fit_requirements": "structural extractor fit-free; scaler and baseline vocabulary fit on fold-training only",
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for metric in ("precision", "recall", "f1", "pr_auc", "balanced_accuracy"):
        values = np.asarray([run[metric] for run in runs], dtype=float)
        result[f"cv_{metric}_mean"] = float(values.mean())
        result[f"cv_{metric}_std"] = float(values.std(ddof=0))
    return result


def _select(records: list[dict[str, Any]], floor: float) -> tuple[dict[str, Any], bool]:
    eligible = [r for r in records if r["cv_precision_mean"] >= floor]
    pool = eligible or records
    selected = max(pool, key=lambda r: (r["cv_f1_mean"], r["cv_recall_mean"], r["cv_balanced_accuracy_mean"], -r["structural_dimension"], r["representation_id"]))
    return selected, not bool(eligible)


def _evaluate(family: str, parameters: dict[str, Any], seed: int, train_x: Any, train_y: np.ndarray, later_x: Any, later_y: np.ndarray) -> tuple[dict[str, Any], float]:
    started = time.perf_counter(); model = build_baseline_model(family, parameters, seed)
    model.fit(train_x, train_y); predictions = np.asarray(model.predict(later_x), dtype=int)
    metrics = binary_classification_metrics(later_y, predictions, positive_class_scores(model, later_x))
    return metrics, time.perf_counter() - started


def run_phase6d1(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter(); seed = int(config["seed"]); set_random_seed(seed)
    manifests = {name: verify_frozen_manifest(config[f"frozen_{name}_manifest"]) for name in ("phase55", "phase6a", "phase6b", "phase6c")}
    partitions = load_development_data(config["dataset"], config["split_membership"])
    training, validation = partitions["train"], partitions["validation"]
    indexed = training.set_index("row_index", drop=False)
    folds = make_positive_occurrence_folds(training, config["temporal_cv"]["train_positive_ends"], config["temporal_cv"]["validation_positive_ends"])
    fold_definitions = [fold_metadata(fold, indexed) for fold in folds]
    feature_config = yaml.safe_load(config["paper_feature_config"].read_text(encoding="utf-8")); min_df = int(feature_config["ngram"]["min_df"])
    frozen = load_baseline_config(config["paper_model_config"])["models"]
    parameters = {family: baseline_parameters(family, frozen[family]) for family in ("ridge", "svm", "xgboost")}
    common = {"hypothesis": config["hypothesis"], "git_commit": _git_commit(), "random_seed": seed,
              "dataset_sha256": sha256_file(config["dataset"]), "split_config_sha256": manifests["phase55"]["configs/paper_aligned_split.yaml"],
              "feature_config_sha256": manifests["phase55"]["configs/paper_aligned_features.yaml"], "experiment_config_sha256": config["config_sha256"]}
    records: list[dict[str, Any]] = []; selected_by_family = {}; fallbacks = {}
    for family in ("ridge", "svm", "xgboost"):
        family_records = []
        fold_cache = {}
        for fold in folds:
            train = indexed.loc[list(fold.train_indices)].reset_index(drop=True); later = indexed.loc[list(fold.validation_indices)].reset_index(drop=True)
            fold_cache[fold.fold] = (
                train, later,
                build_feature_pair(config["baseline_features"][family], train, later, ngram_min_df=min_df)[:2],
                (transform_structural_features(train["opcode"]), transform_structural_features(later["opcode"])),
                _creator_pair(train, later),
            )
        for specification in config["representations"]:
            runs = []
            for fold in folds:
                train, later, baseline, structural, creator = fold_cache[fold.fold]
                train_x, later_x, audit = build_representation_pair(specification, family, train, later, baseline, structural, creator)
                metrics, elapsed = _evaluate(family, parameters[family], seed, train_x, train["label"].to_numpy(int), later_x, later["label"].to_numpy(int))
                meta = fold_definitions[fold.fold - 1]
                runs.append({"fold_id": fold.fold, "model_family": family, "representation_id": specification["id"],
                             "train_block_range": [meta["train_block_min"], meta["train_block_max"]], "cv_block_range": [meta["validation_block_min"], meta["validation_block_max"]],
                             **audit, **metrics, "runtime_seconds": elapsed})
            record = {"experiment_id": f"phase6d1_{family}_{specification['id']}", "model_family": family,
                      "representation_id": specification["id"], **common, **runs[0], "temporal_cv_fold_metrics": runs, **_aggregate(runs),
                      "selected_by_temporal_cv": False, "selection_reason": None}
            records.append(record); family_records.append(record)
        selected, fallback = _select(family_records, float(config["selection"]["minimum_cv_precision"]))
        selected["selected_by_temporal_cv"] = True
        selected["selection_reason"] = "maximum mean temporal-CV F1 subject to precision >= 0.80; ties recall, balanced accuracy, fewer structural features" if not fallback else "diagnostic fallback: no representation met mean temporal-CV precision >= 0.80"
        for record in family_records:
            if not record["selected_by_temporal_cv"]: record["selection_reason"] = "not selected by predefined temporal-CV criterion"
        selected_by_family[family] = selected["representation_id"]; fallbacks[family] = fallback
    validations = []
    for selected in [r for r in records if r["selected_by_temporal_cv"]]:
        family = selected["model_family"]; specification = next(r for r in config["representations"] if r["id"] == selected["representation_id"])
        baseline = build_feature_pair(config["baseline_features"][family], training, validation, ngram_min_df=min_df)[:2]
        structural = (transform_structural_features(training["opcode"]), transform_structural_features(validation["opcode"]))
        creator = _creator_pair(training, validation)
        train_x, validation_x, audit = build_representation_pair(specification, family, training, validation, baseline, structural, creator)
        metrics, elapsed = _evaluate(family, parameters[family], seed, train_x, training["label"].to_numpy(int), validation_x, validation["label"].to_numpy(int))
        reference = config["reference_validation"]
        deltas = {key: metrics[key] - reference[key] for key in ("precision", "recall", "f1", "pr_auc", "balanced_accuracy", "tn", "fp", "fn", "tp")}
        validations.append({"model_family": family, "representation_id": selected["representation_id"], **audit,
                            "frozen_baseline_metrics": reference, "structural_candidate_metrics": metrics, "metric_deltas": deltas, "runtime_seconds": elapsed})
    supported = any(v["structural_candidate_metrics"]["precision"] >= .80 and v["structural_candidate_metrics"]["f1"] > float(config["reference_validation"]["f1"]) for v in validations)
    return {"generated_at_utc": datetime.now(UTC).isoformat(), "phase": "6D1", "hypothesis": config["hypothesis"],
            "protocol": {"intervention": "contract-local opcode-name structural features only", "creator_feature": "included in every representation with chronological leakage-safe semantics", "sampling": "none", "class_weighting": "none", "decision_thresholds": "native defaults", "final_test_usage": "prohibited", "mulcas_unchanged": True},
            "provenance": {**common, "frozen_manifests": manifests}, "feature_schema": {"groups": {g: list(FEATURE_GROUPS[g]) for g in GROUPS}, "ordered_feature_names": list(FEATURE_NAMES), "total_features": len(FEATURE_NAMES), "bytecode_length": "omitted; opcode audit has no missing rows and Phase 6D1 remains opcode-name-only"},
            "temporal_cv_folds": fold_definitions, "records": records, "selected_by_family": selected_by_family,
            "cv_precision_constraint_fallback": fallbacks, "validation_records": validations, "reference_validation": config["reference_validation"], "h6d1_supported": supported,
            "total_runtime_seconds": time.perf_counter() - started}


def save_results(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / name for key, name in {"json":"phase6d1_results.json","cv":"phase6d1_cv_runs.csv","candidates":"phase6d1_candidates.csv","validation":"phase6d1_validation.csv","schema":"phase6d1_feature_schema.json","summary":"phase6d1_feature_summary.csv","report":"phase6d1_report.md","folds":"temporal_cv_folds.json"}.items()}
    paths["json"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame([run for record in results["records"] for run in record["temporal_cv_fold_metrics"]]).to_csv(paths["cv"], index=False)
    pd.DataFrame([{k:v for k,v in record.items() if k != "temporal_cv_fold_metrics"} for record in results["records"]]).to_csv(paths["candidates"], index=False)
    flat_validation=[]
    for record in results["validation_records"]:
        flat={k:v for k,v in record.items() if k not in {"frozen_baseline_metrics","structural_candidate_metrics","metric_deltas"}}
        for prefix,key in (("baseline","frozen_baseline_metrics"),("candidate","structural_candidate_metrics"),("delta","metric_deltas")): flat.update({f"{prefix}_{name}":value for name,value in record[key].items()})
        flat_validation.append(flat)
    pd.DataFrame(flat_validation).to_csv(paths["validation"], index=False)
    paths["schema"].write_text(json.dumps(results["feature_schema"], indent=2), encoding="utf-8")
    pd.DataFrame([{"group": group, "feature_count": len(features), "feature_names": json.dumps(features)} for group, features in results["feature_schema"]["groups"].items()]).to_csv(paths["summary"], index=False)
    paths["folds"].write_text(json.dumps(results["temporal_cv_folds"], indent=2), encoding="utf-8")
    rows=["| Model | Selected representation | CV precision | CV recall | CV F1 | Validation precision | Validation recall | Validation F1 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    by_model={v["model_family"]:v for v in results["validation_records"]}
    for r in results["records"]:
        if r["selected_by_temporal_cv"]:
            m=by_model[r["model_family"]]["structural_candidate_metrics"]; rows.append(f"| {r['model_family']} | {r['representation_id']} | {r['cv_precision_mean']:.6f} | {r['cv_recall_mean']:.6f} | {r['cv_f1_mean']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} |")
    paths["report"].write_text("# Phase 6D1: Structural Static Features\n\nCreation-time, opcode-name-only features were evaluated with frozen models, native thresholds, no sampling, and no class weighting. Final-test rows were not loaded.\n\n"+"\n".join(rows)+f"\n\n- Added structural features: {results['feature_schema']['total_features']}\n- H6D1 supported: `{str(results['h6d1_supported']).lower()}`\n", encoding="utf-8")
    return paths


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH); args=parser.parse_args(); config=load_config(args.config)
    for name,path in save_results(run_phase6d1(config), config["output_dir"]).items(): print(f"Wrote {name}: {path}")


if __name__ == "__main__": main()
