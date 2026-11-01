"""Forward-only score fusion of frozen MulCas and GIN predictions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import sklearn
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/extension/phase8r3b.yaml"
PERIODS = ("P2", "P3", "P4")
EXPECTED_PERIOD_ROWS = {"P2": 5141, "P3": 289, "P4": 150}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["inputs"]["aligned_predictions"] = _resolve(
        config["inputs"]["aligned_predictions"]
    )
    config["frozen_manifests"] = [
        _resolve(value) for value in config["frozen_manifests"]
    ]
    config["output_dir"] = _resolve(config["output_dir"])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["periods"] != list(PERIODS) or config["prohibited_period"] != "P5":
        raise ValueError("Phase 8R3B is restricted to P2-P4.")
    if config["inputs"]["gin_seeds"] != [7, 42, 123, 2026, 999]:
        raise ValueError("The frozen five-seed GIN policy changed.")
    model = config["meta_classifier"]
    expected = {
        "family": "LogisticRegression", "penalty": "l2", "C": 1.0,
        "class_weight": None, "solver": "liblinear", "max_iter": 2000,
        "threshold": 0.5, "scaler": "StandardScaler",
    }
    if model != expected:
        raise ValueError("The fixed Phase 8R3B meta-classifier configuration changed.")
    selection = config["selection"]
    confirmation = config["confirmation"]
    if selection["meta_train_periods"] != ["P2"] or selection["selection_period"] != "P3":
        raise ValueError("Feature-set selection must use P2 -> P3 only.")
    if confirmation["meta_train_periods"] != ["P2", "P3"] or confirmation["evaluation_period"] != "P4":
        raise ValueError("Forward confirmation must use P2+P3 -> P4 only.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in config["frozen_manifests"]:
        observed.update(verify_frozen_manifest(manifest))
    return observed


def load_frozen_predictions(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(config["inputs"]["aligned_predictions"])
    if set(frame["period"]) != set(PERIODS) or "P5" in set(frame["period"]):
        raise AssertionError("Frozen input must contain exactly P2-P4 and never P5.")
    if frame.groupby("period").size().to_dict() != EXPECTED_PERIOD_ROWS:
        raise AssertionError("Frozen Phase 7 period memberships changed.")
    if frame["row_index"].duplicated().any():
        raise AssertionError("Frozen row identities must be unique.")
    seed_columns = [
        f"gin_probability_seed_{seed}" for seed in config["inputs"]["gin_seeds"]
    ]
    expected_mean = frame[seed_columns].mean(axis=1)
    np.testing.assert_allclose(
        frame["gin_mean_probability"], expected_mean, rtol=0, atol=1e-15
    )
    frame = frame.sort_values("row_index", kind="mergesort").reset_index(drop=True)
    frame["absolute_score_disagreement"] = np.abs(
        frame["mulcas_score"] - frame["gin_mean_probability"]
    )
    frame["route_first_viewer"] = frame["cascade_stage"].eq(1).astype(int)
    frame["route_second_viewer"] = frame["cascade_stage"].eq(2).astype(int)
    frame["route_voting_stage"] = frame["reached_voting_stage"].astype(bool).astype(int)
    if not frame[["route_first_viewer", "route_second_viewer", "route_voting_stage"]].sum(axis=1).eq(1).all():
        raise AssertionError("Every row must have exactly one frozen MulCas route.")
    frame["historically_novel_code"] = (
        ~frame["has_any_exact_historical_clone"].astype(bool)
    ).astype(int)
    frame["unseen_creator"] = (~frame["previously_seen_creator"].astype(bool)).astype(int)
    frame["creator_history_feature"] = frame["creator_feature_value"].astype(int)
    return frame


def feature_schemas(config: dict[str, Any]) -> dict[str, list[str]]:
    schemas = {
        name: list(columns) for name, columns in config["feature_sets"].items()
    }
    if set(schemas) != {"hybrid_a", "hybrid_b"}:
        raise ValueError("Exactly Hybrid A and Hybrid B must be evaluated.")
    if schemas["hybrid_b"][: len(schemas["hybrid_a"])] != schemas["hybrid_a"]:
        raise ValueError("Hybrid B must extend the frozen Hybrid A score schema.")
    return schemas


def _make_model(config: dict[str, Any]) -> LogisticRegression:
    model = config["meta_classifier"]
    return LogisticRegression(
        l1_ratio=0, C=float(model["C"]),
        class_weight=model["class_weight"], solver=model["solver"],
        max_iter=int(model["max_iter"]), random_state=int(config["random_seed"]),
    )


def runtime_preflight(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    columns = feature_schemas(config)["hybrid_a"]
    train = frame[frame["period"].eq("P2")]
    x = train[columns].to_numpy(dtype=float)
    y = train["label"].to_numpy(dtype=int)
    started = time.perf_counter()
    scaler = StandardScaler().fit(x)
    model = _make_model(config).fit(scaler.transform(x), y)
    benchmark_seconds = time.perf_counter() - started
    estimate = max(
        float(config["preflight"]["minimum_estimate_seconds"]),
        benchmark_seconds * float(config["preflight"]["conservative_fit_multiplier"]),
    )
    result = {
        "cpu_model": platform.processor(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "system_ram_bytes": int(psutil.virtual_memory().total),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "execution_device": "cpu",
        "sklearn_version": sklearn.__version__,
        "benchmark": {
            "classifier": "LogisticRegression", "feature_set": "hybrid_a",
            "training_periods": ["P2"], "row_count": len(train),
            "feature_count": len(columns), "elapsed_seconds": benchmark_seconds,
            "iterations": [int(value) for value in model.n_iter_],
        },
        "estimated_total_runtime_seconds": estimate,
        "stop_limit_seconds": float(config["preflight"]["maximum_estimated_runtime_seconds"]),
        "full_experiment_authorized": estimate <= float(config["preflight"]["maximum_estimated_runtime_seconds"]),
    }
    return result


def _fit_predict(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_train = train[columns].to_numpy(dtype=float)
    x_eval = evaluation[columns].to_numpy(dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_eval).all():
        raise ValueError("Meta features must be finite.")
    scaler = StandardScaler().fit(x_train)
    model = _make_model(config)
    started = time.perf_counter()
    model.fit(scaler.transform(x_train), train["label"].to_numpy(dtype=int))
    fit_seconds = time.perf_counter() - started
    probability = model.predict_proba(scaler.transform(x_eval))[:, 1]
    threshold = float(config["meta_classifier"]["threshold"])
    prediction = (probability >= threshold).astype(int)
    diagnostics = {
        "fit_seconds": fit_seconds,
        "converged": bool(np.all(model.n_iter_ < int(config["meta_classifier"]["max_iter"]))),
        "iterations": [int(value) for value in model.n_iter_],
        "scaler_fit_row_indices": [int(value) for value in train["row_index"]],
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(), "intercept": model.intercept_.tolist(),
    }
    return probability, prediction, diagnostics


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    return binary_classification_metrics(
        frame["label"].to_numpy(dtype=int), prediction.astype(int), score.astype(float)
    )


def _component_metrics(frame: pd.DataFrame, period: str) -> list[dict[str, Any]]:
    selected = frame[frame["period"].eq(period)]
    rows = []
    for system, prediction, score in (
        ("frozen_mulcas", selected["mulcas_prediction"], selected["mulcas_score"]),
        ("frozen_gin_ensemble", selected["gin_ensemble_prediction"], selected["gin_mean_probability"]),
    ):
        rows.append({
            "period": period, "system": system, "feature_set": "frozen_component",
            "eligible": None, "selection_role": "comparison_only",
            **_metrics(selected, prediction.to_numpy(), score.to_numpy()),
        })
    return rows


def select_candidate(candidate_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    floor = float(config["selection"]["precision_floor"])
    candidates = [row for row in candidate_rows if row["system"].startswith("hybrid_")]
    for row in candidates:
        row["eligible"] = bool(row["precision"] >= floor)
    eligible = [row for row in candidates if row["eligible"]]
    pool = eligible or candidates
    simplicity = {"hybrid_a": 1, "hybrid_b": 0}
    selected = max(
        pool,
        key=lambda row: (
            row["f1"], row["recall"], row["balanced_accuracy"],
            simplicity[row["system"]],
        ),
    )
    return {
        "feature_set": selected["system"],
        "eligible": bool(selected["eligible"]),
        "selection_period": "P3",
        "selection_reason": (
            "Highest ranked eligible P3 candidate under the predefined rule."
            if selected["eligible"] else
            "No candidate met the P3 precision floor; selected as a diagnostic fallback."
        ),
    }


def _prediction_export(
    evaluation: pd.DataFrame, system: str, score: np.ndarray, prediction: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "row_index", "address", "period", "createdBlockNumber", "label",
        "mulcas_prediction", "mulcas_score", "gin_ensemble_prediction",
        "gin_mean_probability", "accepted_viewer", "cascade_stage",
        "reached_voting_stage", "historically_novel_code", "unseen_creator",
        "creator_history_feature",
    ]
    result = evaluation[columns].copy()
    result.insert(0, "hybrid_system", system)
    result["hybrid_probability"] = score
    result["hybrid_prediction"] = prediction
    return result


def error_transitions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (system, period), group in predictions.groupby(["hybrid_system", "period"], sort=True):
        label = group["label"]
        mulcas = group["mulcas_prediction"]
        gin = group["gin_ensemble_prediction"]
        hybrid = group["hybrid_prediction"]
        measures = {
            "mulcas_fn_detected_by_gin_recovered": label.eq(1) & mulcas.eq(0) & gin.eq(1) & hybrid.eq(1),
            "mulcas_fp_rejected_by_gin_suppressed": label.eq(0) & mulcas.eq(1) & gin.eq(0) & hybrid.eq(0),
            "new_false_positive_vs_mulcas": label.eq(0) & mulcas.eq(0) & hybrid.eq(1),
            "new_false_negative_vs_mulcas": label.eq(1) & mulcas.eq(1) & hybrid.eq(0),
            "both_components_wrong_hybrid_correct": mulcas.ne(label) & gin.ne(label) & hybrid.eq(label),
        }
        for transition, mask in measures.items():
            rows.append({"system": system, "period": period, "transition": transition, "count": int(mask.sum())})
    return pd.DataFrame(rows)


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = {
        "historically_novel_ponzi": lambda x: x["label"].eq(1) & x["historically_novel_code"].eq(1),
        "historical_clone_ponzi": lambda x: x["label"].eq(1) & x["historically_novel_code"].eq(0),
        "unseen_creator_ponzi": lambda x: x["label"].eq(1) & x["unseen_creator"].eq(1),
        "seen_creator_ponzi": lambda x: x["label"].eq(1) & x["unseen_creator"].eq(0),
    }
    for (system, period), frame in predictions.groupby(["hybrid_system", "period"], sort=True):
        for subgroup, selector in groups.items():
            selected = frame[selector(frame)]
            support = len(selected)
            hybrid_tp = int(selected["hybrid_prediction"].eq(1).sum())
            mulcas_tp = int(selected["mulcas_prediction"].eq(1).sum())
            gin_tp = int(selected["gin_ensemble_prediction"].eq(1).sum())
            rows.append({
                "system": system, "period": period, "subgroup": subgroup,
                "positive_support": support,
                "hybrid_true_positive": hybrid_tp,
                "hybrid_false_negative": support - hybrid_tp,
                "hybrid_recall": hybrid_tp / support if support else None,
                "frozen_mulcas_true_positive": mulcas_tp,
                "frozen_mulcas_recall": mulcas_tp / support if support else None,
                "frozen_gin_true_positive": gin_tp,
                "frozen_gin_recall": gin_tp / support if support else None,
            })
    return pd.DataFrame(rows)


def _build_report(
    p3: pd.DataFrame, p4: pd.DataFrame, selection: dict[str, Any],
    transitions: pd.DataFrame, subgroups: pd.DataFrame, preflight: dict[str, Any],
    supported: bool, runtime: float,
) -> str:
    def metric_rows(frame: pd.DataFrame) -> list[str]:
        return [
            f"| {row.system} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | "
            f"{row.pr_auc:.4f} | {row.balanced_accuracy:.4f} | "
            f"{int(row.tn)}/{int(row.fp)}/{int(row.fn)}/{int(row.tp)} |"
            for row in frame.itertuples(index=False)
        ]
    subgroup_rows = [
        f"| {row.system} | {row.period} | {row.subgroup} | {int(row.positive_support)} | "
        f"{('N/A' if pd.isna(row.frozen_mulcas_recall) else f'{row.frozen_mulcas_recall:.4f}')} | "
        f"{('N/A' if pd.isna(row.frozen_gin_recall) else f'{row.frozen_gin_recall:.4f}')} | "
        f"{('N/A' if pd.isna(row.hybrid_recall) else f'{row.hybrid_recall:.4f}')} |"
        for row in subgroups.itertuples(index=False)
    ]
    shared = transitions[transitions["transition"].eq("both_components_wrong_hybrid_correct")]
    shared_rows = [
        f"- {row.system} on {row.period}: {int(row.count)}"
        for row in shared.itertuples(index=False)
    ]
    return "\n".join([
        "# Phase 8R3B: Forward MulCas-GIN Score Fusion", "",
        "MulCas and GIN were not retrained. The meta-classifier used P2 for feature-set selection on P3, then the selected schema was frozen and refitted on P2+P3 for one P4 confirmation. P5 was not loaded.", "",
        "## Score Semantics", "",
        "`mulcas_score` is the frozen accepted viewer's positive-class score for cascade decisions. For voting rows it is the maximum frozen voting-viewer score. It is a deployment-time confidence proxy, not a route-independent calibrated probability. Hybrid B therefore includes route one-hot indicators.", "",
        "## Runtime Preflight", "",
        f"CPU execution was used (`torch {preflight['torch_version']}`, CUDA={preflight['cuda_available']}). The representative fit took {preflight['benchmark']['elapsed_seconds']:.4f}s; estimated full runtime was {preflight['estimated_total_runtime_seconds']:.2f}s and actual experiment runtime was {runtime:.4f}s.", "",
        "## P3 Selection", "", "| System | Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN/FP/FN/TP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |", *metric_rows(p3), "",
        f"Selected: **{selection['feature_set']}**. Eligible: **{selection['eligible']}**. {selection['selection_reason']}", "",
        "## P4 Forward Confirmation", "", "| System | Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN/FP/FN/TP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |", *metric_rows(p4), "",
        "## Novelty And Creator Subgroups", "", "| System | Period | Subgroup | Positive support | MulCas recall | GIN recall | Hybrid recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |", *subgroup_rows, "",
        "## Shared-Error Corrections", "", *shared_rows, "",
        "Counts are reported separately because the two P3 candidates evaluate the same rows; they must not be summed as distinct contracts.", "",
        "## Conclusion", "",
        f"H8R3B is **{'supported' if supported else 'not supported'}** under the predeclared P3 eligibility and P4 confirmation rule.", "",
    ])


def run(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    set_random_seed(int(config["random_seed"]))
    frozen = verify_frozen_inputs(config)
    frame = load_frozen_predictions(config)
    schemas = feature_schemas(config)
    preflight = runtime_preflight(frame, config)
    if not preflight["full_experiment_authorized"]:
        raise RuntimeError("Estimated Phase 8R3B runtime exceeds 30 minutes.")

    p2 = frame[frame["period"].eq("P2")]
    p3 = frame[frame["period"].eq("P3")]
    candidate_rows = _component_metrics(frame, "P3")
    prediction_frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}
    for name, columns in schemas.items():
        probability, prediction, detail = _fit_predict(p2, p3, columns, config)
        candidate_rows.append({
            "period": "P3", "system": name, "feature_set": name,
            "eligible": None, "selection_role": "candidate",
            **_metrics(p3, prediction, probability),
        })
        diagnostics[f"p2_to_p3_{name}"] = detail
        prediction_frames.append(_prediction_export(p3, name, probability, prediction))

    selection = select_candidate(candidate_rows, config)
    p3_metrics = pd.DataFrame(candidate_rows)
    for row in candidate_rows:
        if row["system"] == selection["feature_set"]:
            row["selection_role"] = "selected" if selection["eligible"] else "diagnostic_fallback"
    p3_metrics = pd.DataFrame(candidate_rows)

    p2_p3 = frame[frame["period"].isin(["P2", "P3"])]
    p4 = frame[frame["period"].eq("P4")]
    selected_name = selection["feature_set"]
    probability, prediction, detail = _fit_predict(
        p2_p3, p4, schemas[selected_name], config
    )
    diagnostics[f"p2_p3_to_p4_{selected_name}"] = detail
    p4_rows = _component_metrics(frame, "P4")
    p4_rows.append({
        "period": "P4", "system": selected_name, "feature_set": selected_name,
        "eligible": selection["eligible"], "selection_role": "forward_confirmation",
        **_metrics(p4, prediction, probability),
    })
    p4_metrics = pd.DataFrame(p4_rows)
    prediction_frames.append(_prediction_export(p4, selected_name, probability, prediction))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    transitions = error_transitions(predictions)
    subgroups = subgroup_metrics(predictions)

    selected_p4 = p4_metrics[p4_metrics["system"].eq(selected_name)].iloc[0]
    frozen_p4 = p4_metrics[p4_metrics["system"].isin(["frozen_mulcas", "frozen_gin_ensemble"])]
    support_rule = config["hypothesis_support"]
    supported = bool(
        selection["eligible"]
        and selected_p4["precision"] >= float(support_rule["p4_precision_floor"])
        and selected_p4["f1"] > float(frozen_p4["f1"].max())
    )
    for metrics_frame in (p3_metrics, p4_metrics):
        mulcas = metrics_frame[metrics_frame["system"].eq("frozen_mulcas")].iloc[0]
        gin = metrics_frame[metrics_frame["system"].eq("frozen_gin_ensemble")].iloc[0]
        for metric in ("precision", "recall", "f1", "pr_auc", "balanced_accuracy"):
            metrics_frame[f"delta_{metric}_vs_mulcas"] = metrics_frame[metric] - mulcas[metric]
            metrics_frame[f"delta_{metric}_vs_gin"] = metrics_frame[metric] - gin[metric]

    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    actual_runtime = time.perf_counter() - started
    preflight["actual_experiment_runtime_seconds"] = actual_runtime
    schema_record = {
        "score_semantics": {
            "cascade_rows": "accepted frozen viewer positive-class score",
            "voting_rows": "maximum positive-class score among frozen voting viewers",
            "calibrated_probability": False,
        },
        "feature_sets": schemas,
        "route_categories": ["first_viewer", "second_viewer", "voting_stage"],
        "scaling": "StandardScaler fitted only on meta-training rows",
    }
    results = {
        "phase": "8R3B", "hypothesis": config["hypothesis"],
        "base_models_retrained": False, "threshold_tuned": False,
        "p5_loaded_or_evaluated": False, "gin_ensemble": "arithmetic_mean_of_five_frozen_seeds",
        "random_seed": int(config["random_seed"]), "frozen_artifact_hashes": frozen,
        "config_sha256": config["config_sha256"], "preflight": preflight,
        "meta_classifier": config["meta_classifier"], "selection": selection,
        "p3_candidate_metrics": p3_metrics.to_dict(orient="records"),
        "p4_confirmation_metrics": p4_metrics.to_dict(orient="records"),
        "training_diagnostics": diagnostics, "h8r3b_supported": supported,
    }
    p3_metrics.to_csv(output / "p3_candidate_metrics.csv", index=False)
    p4_metrics.to_csv(output / "p4_confirmation_metrics.csv", index=False)
    predictions.to_csv(output / "hybrid_predictions.csv", index=False)
    transitions.to_csv(output / "error_transition.csv", index=False)
    subgroups.to_csv(output / "subgroup_metrics.csv", index=False)
    (output / "runtime_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    (output / "meta_feature_schema.json").write_text(json.dumps(schema_record, indent=2), encoding="utf-8")
    (output / "phase8r3b_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "phase8r3b_report.md").write_text(
        _build_report(p3_metrics, p4_metrics, selection, transitions, subgroups, preflight, supported, actual_runtime),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    result = run(load_config(args.config))
    print(json.dumps({
        "selected_feature_set": result["selection"]["feature_set"],
        "eligible": result["selection"]["eligible"],
        "h8r3b_supported": result["h8r3b_supported"],
    }, indent=2))


if __name__ == "__main__":
    main()
