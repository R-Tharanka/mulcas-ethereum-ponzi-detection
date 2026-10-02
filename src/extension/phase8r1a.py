"""Controlled recency-aware temporal backtesting for Phase 8R1A."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.experiments.phase7a import (
    _load_frozen_configs,
    _period_summary,
    _viewer_output,
    build_historical_feature_matrices,
)
from src.models.baselines import build_baseline_model
from src.models.mulcas import MulCasCascade, ViewerPrediction, ViewerSpec
from src.models.run_mulcas import positive_probability
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "extension" / "phase8r1a.yaml"
PREDICTION_PERIODS = ("P2", "P3", "P4")
METRIC_NAMES = ("precision", "recall", "f1", "balanced_accuracy", "pr_auc")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset",
        "phase7a_config",
        "phase7a_period_definitions",
        "phase7a_results",
        "phase7a_predictions",
        "phase7e_clone_membership",
        "paper_feature_config",
        "paper_model_config",
        "paper_mulcas_config",
        "frozen_phase7_manifest",
        "frozen_phase8r0_manifest",
        "output_dir",
    ):
        config[key] = _resolve(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if tuple(config["periods"]["allowed_prediction"]) != PREDICTION_PERIODS:
        raise ValueError("Phase 8R1A must predict P2-P4 exactly.")
    if config["selection"]["p5_usage"] != "prohibited":
        raise ValueError("P5 use must be prohibited.")
    expected_alphas = (0.25, 0.50, 0.75)
    observed_alphas = tuple(
        float(item["alpha"])
        for item in config["strategies"]
        if item["family"] == "exponential_decay"
    )
    if observed_alphas != expected_alphas:
        raise ValueError("Exponential-decay candidates must be 0.25, 0.50, 0.75.")
    if config["sample_weight"]["class_weight"] is not None:
        raise ValueError("Class weighting is prohibited.")
    if config["sample_weight"]["sampling"] != "none":
        raise ValueError("Sampling is prohibited.")
    return config


def verify_frozen_chain(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in (config["frozen_phase7_manifest"], config["frozen_phase8r0_manifest"]):
        for relative, digest in verify_frozen_manifest(manifest).items():
            if relative in observed and observed[relative] != digest:
                raise RuntimeError(f"Conflicting frozen hash for {relative}")
            observed[relative] = digest
    return observed


def load_p0_p4_membership(config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Read exactly the frozen P0-P4 prefix; no P5 row is loaded."""
    definitions = json.loads(config["phase7a_period_definitions"].read_text(encoding="utf-8"))
    selected = [item for item in definitions if item["period"] != "P5"]
    if tuple(item["period"] for item in selected) != ("P0", "P1", "P2", "P3", "P4"):
        raise ValueError("Frozen Phase 7 period definitions are incomplete.")
    row_limit = sum(int(item["row_count"]) for item in selected)
    frame = pd.read_csv(
        config["dataset"],
        usecols=["address", "opcode", "creator", "createdBlockNumber", "label"],
        nrows=row_limit,
        low_memory=False,
    )
    frame["row_index"] = frame.index.astype(int)
    frame["opcode"] = frame["opcode"].fillna("").astype(str)
    if len(frame) != row_limit:
        raise AssertionError("Could not read the exact frozen P0-P4 row prefix.")
    if frame[["address", "creator", "createdBlockNumber", "label"]].isna().any().any():
        raise ValueError("Required temporal fields contain null values.")

    frame["period"] = ""
    for definition in selected:
        mask = frame["createdBlockNumber"].between(
            int(definition["block_min"]), int(definition["block_max"]), inclusive="both"
        )
        frame.loc[mask, "period"] = definition["period"]
        rows = frame.loc[mask]
        if len(rows) != int(definition["row_count"]) or int(rows["label"].sum()) != int(definition["positive_count"]):
            raise AssertionError(f"Frozen membership mismatch for {definition['period']}.")
    if (frame["period"] == "").any() or "P5" in set(frame["period"]):
        raise AssertionError("P0-P4 loader encountered an unassigned or P5 row.")

    prediction_row_limit = sum(
        int(item["row_count"]) for item in selected if item["period"] in PREDICTION_PERIODS
    )
    phase7_predictions = pd.read_csv(
        config["phase7a_predictions"],
        usecols=["row_index", "period"],
        nrows=prediction_row_limit,
    )
    if not set(phase7_predictions["period"]).issubset(PREDICTION_PERIODS):
        raise AssertionError("Phase 7 membership check loaded a prohibited period.")
    expected = set(phase7_predictions["row_index"].astype(int))
    observed = set(frame.loc[frame["period"].isin(PREDICTION_PERIODS), "row_index"].astype(int))
    if observed != expected:
        raise AssertionError("Phase 8R1A did not reuse exact Phase 7 P2-P4 memberships.")
    return frame.sort_values(["createdBlockNumber", "row_index"], kind="mergesort"), selected


def strategy_training_periods(
    strategy: dict[str, Any], prediction_period: str
) -> tuple[tuple[str, ...], bool]:
    target_number = int(prediction_period[1:])
    available = tuple(f"P{number}" for number in range(target_number))
    if strategy["family"] == "hard_window":
        requested = int(strategy["window_periods"])
        return available[-requested:], len(available) < requested
    return available, False


def period_sample_weights(
    frame: pd.DataFrame, strategy: dict[str, Any], prediction_period: str
) -> np.ndarray | None:
    """Return estimator-only temporal weights; controls/windows remain unweighted."""
    if strategy["family"] != "exponential_decay":
        return None
    alpha = float(strategy["alpha"])
    newest = int(prediction_period[1:]) - 1
    ages = newest - frame["period"].str[1:].astype(int).to_numpy()
    if np.any(ages < 0):
        raise AssertionError("Temporal weights received future rows.")
    return np.power(alpha, ages, dtype=float)


def fit_estimator(
    model: Any,
    features: csr_matrix,
    labels: np.ndarray,
    sample_weight: np.ndarray | None,
) -> Any:
    """Apply temporal sample weights to estimator fitting only."""
    if sample_weight is None:
        return model.fit(features, labels)
    return model.fit(features, labels, sample_weight=sample_weight)


def fit_predict_mulcas(
    train: pd.DataFrame,
    target: pd.DataFrame,
    matrices: dict[str, tuple[csr_matrix, csr_matrix]],
    model_config: dict[str, Any],
    mulcas_config: dict[str, Any],
    *,
    seed: int,
    sample_weight: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, dict[str, ViewerPrediction]]]:
    """Run frozen MulCas, changing only optional estimator row weights."""
    labels = train["label"].to_numpy(dtype=int)
    target_indices = target["row_index"].to_numpy(dtype=int)
    cascade_outputs: dict[str, ViewerPrediction] = {}
    voting_outputs: dict[str, ViewerPrediction] = {}
    training_trace: list[dict[str, Any]] = []
    residual_positions = np.arange(len(train), dtype=int)
    cascade_viewers = mulcas_config["viewer_sequence"][: int(mulcas_config["complexity_control_n"])]

    for stage, viewer in enumerate(cascade_viewers, start=1):
        train_matrix, target_matrix = matrices[viewer["feature_view"]]
        model = build_baseline_model(viewer["model"], model_config["models"][viewer["model"]], seed)
        residual_weights = None if sample_weight is None else sample_weight[residual_positions]
        fit_estimator(model, train_matrix[residual_positions], labels[residual_positions], residual_weights)
        train_probabilities = positive_probability(model, train_matrix[residual_positions])
        confident = (
            train_probabilities < float(viewer["negative_threshold"])
        ) | (train_probabilities > float(viewer["positive_threshold"]))
        training_trace.append({
            "stage": stage,
            "viewer": viewer["name"],
            "training_rows_in": int(len(residual_positions)),
            "positive_training_rows_in": int(labels[residual_positions].sum()),
            "effective_sample_weight_in": float(
                len(residual_positions) if residual_weights is None else residual_weights.sum()
            ),
            "confident_training_rows": int(confident.sum()),
            "training_rows_out": int((~confident).sum()),
            "positive_training_rows_out": int(labels[residual_positions][~confident].sum()),
        })
        residual_positions = residual_positions[~confident]
        cascade_outputs[viewer["name"]] = _viewer_output(model, target_matrix, target_indices)

    for viewer in mulcas_config["viewer_sequence"]:
        train_matrix, target_matrix = matrices[viewer["feature_view"]]
        model = build_baseline_model(viewer["model"], model_config["models"][viewer["model"]], seed)
        fit_estimator(model, train_matrix, labels, sample_weight)
        voting_outputs[viewer["name"]] = _viewer_output(model, target_matrix, target_indices)

    cascade = MulCasCascade(
        cascade_viewers=[
            ViewerSpec(
                name=viewer["name"],
                negative_threshold=float(viewer["negative_threshold"]),
                positive_threshold=float(viewer["positive_threshold"]),
            )
            for viewer in cascade_viewers
        ],
        voting_viewers=[viewer["name"] for viewer in mulcas_config["viewer_sequence"]],
        voting_threshold=float(mulcas_config["voting_threshold"]),
    )
    result = cascade.predict(target_indices, cascade_outputs, voting_outputs)
    return result.predictions, result.traces, training_trace, {
        "cascade": cascade_outputs,
        "voting": voting_outputs,
    }


def evaluate_strategy_period(
    assigned: pd.DataFrame,
    strategy: dict[str, Any],
    prediction_period: str,
    feature_config: dict[str, Any],
    model_config: dict[str, Any],
    mulcas_config: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    training_periods, history_limited = strategy_training_periods(strategy, prediction_period)
    train = assigned[assigned["period"].isin(training_periods)].copy()
    target = assigned[assigned["period"] == prediction_period].copy()
    train = train.sort_values(["createdBlockNumber", "row_index"], kind="mergesort").reset_index(drop=True)
    target = target.sort_values(["createdBlockNumber", "row_index"], kind="mergesort").reset_index(drop=True)
    if train.empty or target.empty:
        raise ValueError(f"Empty partition for {strategy['id']} {prediction_period}.")
    if int(train["createdBlockNumber"].max()) >= int(target["createdBlockNumber"].min()):
        raise AssertionError("Recency backtest is not forward-only.")
    if "P5" in set(train["period"]) | set(target["period"]):
        raise AssertionError("P5 entered Phase 8R1A evaluation.")

    started = time.perf_counter()
    feature_views = list(dict.fromkeys(item["feature_view"] for item in mulcas_config["viewer_sequence"]))
    matrices, feature_audit, creator_state = build_historical_feature_matrices(
        train,
        target,
        feature_views,
        ngram_min_df=int(feature_config["ngram"]["min_df"]),
    )
    weights = period_sample_weights(train, strategy, prediction_period)
    predictions, traces, training_trace, viewer_outputs = fit_predict_mulcas(
        train,
        target,
        matrices,
        model_config,
        mulcas_config,
        seed=seed,
        sample_weight=weights,
    )
    runtime = time.perf_counter() - started

    rows = predictions.rename(columns={
        "sample_index": "row_index",
        "prediction": "final_prediction",
        "probability": "final_score",
        "handled_by": "accepted_viewer",
    }).merge(
        target[["row_index", "address", "createdBlockNumber", "label", "period"]],
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    rows["strategy_id"] = strategy["id"]
    rows["training_periods"] = "+".join(training_periods)
    rows["creator_history_feature"] = creator_state
    rows["reached_voting_stage"] = rows["accepted_viewer"].eq("vote")
    rows["cascade_stage"] = rows["accepted_viewer"].map({
        viewer["name"]: stage
        for stage, viewer in enumerate(
            mulcas_config["viewer_sequence"][: int(mulcas_config["complexity_control_n"])], start=1
        )
    }).fillna(int(mulcas_config["complexity_control_n"]) + 1).astype(int)
    for phase_name, outputs in viewer_outputs.items():
        for viewer_name, output in outputs.items():
            rows[f"{phase_name}__{viewer_name}__score"] = rows["row_index"].map(output.probability)
            rows[f"{phase_name}__{viewer_name}__prediction"] = rows["row_index"].map(output.prediction)

    metrics = binary_classification_metrics(
        rows["label"].to_numpy(dtype=int),
        rows["final_prediction"].to_numpy(dtype=int),
        rows["final_score"].to_numpy(dtype=float),
    )
    routing_counts = {
        str(name): int(count) for name, count in rows["accepted_viewer"].value_counts().items()
    }
    weight_by_period = {
        period: float(1.0 if weights is None else weights[train["period"].eq(period).to_numpy()][0])
        for period in training_periods
    }
    result = {
        "strategy_id": strategy["id"],
        "strategy_family": strategy["family"],
        "alpha": strategy.get("alpha"),
        "prediction_period": prediction_period,
        "training_periods": list(training_periods),
        "history_limited": history_limited,
        "train": _period_summary(train),
        "prediction": _period_summary(target),
        "effective_total_sample_weight": float(len(train) if weights is None else weights.sum()),
        "period_sample_weights": weight_by_period,
        "feature_dimensions": feature_audit["feature_dimensions"],
        "feature_fit_audit": feature_audit,
        "metrics": metrics,
        "routing_counts": routing_counts,
        "cascade_training_trace": training_trace,
        "runtime_seconds": float(runtime),
    }
    trace_rows = traces.rename(columns={"sample_index": "row_index"})
    trace_rows["strategy_id"] = strategy["id"]
    trace_rows["prediction_period"] = prediction_period
    return result, rows, trace_rows


def verify_control_reproduction(
    control_runs: list[dict[str, Any]], phase7a_results_path: Path, tolerance: float
) -> dict[str, Any]:
    reference = json.loads(phase7a_results_path.read_text(encoding="utf-8"))
    expected = {
        item["prediction_period"]: item for item in reference["scenarios"]
        if item["prediction_period"] in PREDICTION_PERIODS
    }
    comparisons: list[dict[str, Any]] = []
    for run in control_runs:
        period = run["prediction_period"]
        metric_deltas = {
            name: float(run["metrics"][name] - expected[period]["metrics"][name])
            for name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"
            )
        }
        confusion_equal = all(
            int(run["metrics"][name]) == int(expected[period]["metrics"][name])
            for name in ("tn", "fp", "fn", "tp")
        )
        routing_equal = run["routing_counts"] == expected[period]["routing_counts"]
        passed = max(abs(value) for value in metric_deltas.values()) <= tolerance and confusion_equal and routing_equal
        comparisons.append({
            "period": period,
            "passed": passed,
            "metric_deltas": metric_deltas,
            "confusion_equal": confusion_equal,
            "routing_equal": routing_equal,
        })
    return {"passed": all(item["passed"] for item in comparisons), "tolerance": tolerance, "comparisons": comparisons}


def _flatten_run(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    return {
        "strategy_id": run["strategy_id"],
        "strategy_family": run["strategy_family"],
        "alpha": run["alpha"],
        "prediction_period": run["prediction_period"],
        "training_periods": "+".join(run["training_periods"]),
        "history_limited": run["history_limited"],
        "training_rows": run["train"]["row_count"],
        "training_positive": run["train"]["positive_count"],
        "training_negative": run["train"]["negative_count"],
        "prediction_rows": run["prediction"]["row_count"],
        "prediction_positive": run["prediction"]["positive_count"],
        "prediction_negative": run["prediction"]["negative_count"],
        "effective_total_sample_weight": run["effective_total_sample_weight"],
        "feature_dimensions": json.dumps(run["feature_dimensions"], sort_keys=True),
        **{name: metrics[name] for name in (
            "accuracy", "balanced_accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "tn", "fp", "fn", "tp"
        )},
        "routing_counts": json.dumps(run["routing_counts"], sort_keys=True),
        "runtime_seconds": run["runtime_seconds"],
    }


def summarize_strategies(
    runs: list[dict[str, Any]], predictions: pd.DataFrame, strategy_order: list[str], precision_floor: float
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for complexity, strategy_id in enumerate(strategy_order):
        selected_runs = [item for item in runs if item["strategy_id"] == strategy_id]
        if len(selected_runs) != 3:
            raise AssertionError(f"Strategy {strategy_id} does not have three temporal runs.")
        values: dict[str, Any] = {}
        for metric in METRIC_NAMES:
            array = np.asarray([item["metrics"][metric] for item in selected_runs], dtype=float)
            values[f"macro_mean_{metric}"] = float(array.mean())
            values[f"macro_std_{metric}"] = float(array.std(ddof=0))
        pooled_rows = predictions[predictions["strategy_id"] == strategy_id]
        pooled = binary_classification_metrics(
            pooled_rows["label"].to_numpy(dtype=int),
            pooled_rows["final_prediction"].to_numpy(dtype=int),
            pooled_rows["final_score"].to_numpy(dtype=float),
        )
        summaries.append({
            "strategy_id": strategy_id,
            **values,
            "eligible": values["macro_mean_precision"] >= precision_floor,
            "complexity_order": complexity,
            **{f"pooled_{key}": value for key, value in pooled.items()},
        })
    return summaries


def select_strategy(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    def ranking(item: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            -item["macro_mean_f1"],
            -item["macro_mean_recall"],
            -item["macro_mean_balanced_accuracy"],
            item["complexity_order"],
        )

    eligible = [item for item in summaries if item["eligible"]]
    selected = sorted(eligible, key=ranking)[0] if eligible else sorted(summaries, key=ranking)[0]
    return {
        "eligible_strategy_count": len(eligible),
        "selected_strategy": selected["strategy_id"] if eligible else None,
        "diagnostic_fallback": None if eligible else selected["strategy_id"],
        "selection_status": "eligible_selection" if eligible else "no_eligible_strategy",
        "selection_reason": (
            "Highest macro P2-P4 F1 among precision-eligible strategies, with predefined tie-breaks."
            if eligible else
            "No strategy met macro mean precision >= 0.80; shown only as the predefined diagnostic fallback."
        ),
    }


def load_subgroup_membership(config: dict[str, Any], definitions: list[dict[str, Any]]) -> pd.DataFrame:
    """Read exactly the P2-P4 prefix from the frozen Phase 7E membership export."""
    row_limit = sum(
        int(item["row_count"]) for item in definitions if item["period"] in PREDICTION_PERIODS
    )
    membership = pd.read_csv(
        config["phase7e_clone_membership"],
        nrows=row_limit,
        usecols=[
            "row_index",
            "period",
            "previously_seen_creator",
            "has_any_exact_historical_clone",
        ],
    )
    if len(membership) != row_limit or not set(membership["period"]).issubset(PREDICTION_PERIODS):
        raise AssertionError("Subgroup loader must contain exact P2-P4 rows only.")
    for column in ("previously_seen_creator", "has_any_exact_historical_clone"):
        membership[column] = membership[column].astype(str).str.lower().map({"true": True, "false": False})
        if membership[column].isna().any():
            raise ValueError(f"Invalid boolean values in {column}.")
    return membership


def subgroup_metrics(predictions: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(membership, on=["row_index", "period"], how="left", validate="many_to_one")
    if merged[["previously_seen_creator", "has_any_exact_historical_clone"]].isna().any().any():
        raise AssertionError("Missing frozen novelty/creator diagnostics.")
    groups = {
        "historically_novel_code": ~merged["has_any_exact_historical_clone"].astype(bool),
        "exact_historical_code_clone": merged["has_any_exact_historical_clone"].astype(bool),
        "unseen_creator": ~merged["previously_seen_creator"].astype(bool),
        "seen_creator": merged["previously_seen_creator"].astype(bool),
    }
    records: list[dict[str, Any]] = []
    for strategy_id in merged["strategy_id"].drop_duplicates():
        for period in PREDICTION_PERIODS:
            base = merged[(merged["strategy_id"] == strategy_id) & (merged["period"] == period)]
            for group_name in groups:
                if group_name == "historically_novel_code":
                    mask = ~base["has_any_exact_historical_clone"].astype(bool)
                elif group_name == "exact_historical_code_clone":
                    mask = base["has_any_exact_historical_clone"].astype(bool)
                elif group_name == "unseen_creator":
                    mask = ~base["previously_seen_creator"].astype(bool)
                else:
                    mask = base["previously_seen_creator"].astype(bool)
                positive = base[mask & base["label"].eq(1)]
                tp = int(positive["final_prediction"].eq(1).sum())
                fn = int(positive["final_prediction"].eq(0).sum())
                records.append({
                    "strategy_id": strategy_id,
                    "period": period,
                    "subgroup": group_name,
                    "positive_count": int(len(positive)),
                    "tp": tp,
                    "fn": fn,
                    "recall": float(tp / len(positive)) if len(positive) else None,
                })
    return pd.DataFrame(records)


def routing_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.groupby(["strategy_id", "period", "accepted_viewer"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = grouped.groupby(["strategy_id", "period"])["count"].transform("sum")
    grouped["proportion"] = grouped["count"] / totals
    control = grouped[grouped["strategy_id"] == "expanding_history"][[
        "period", "accepted_viewer", "count", "proportion"
    ]].rename(columns={"count": "control_count", "proportion": "control_proportion"})
    grouped = grouped.merge(control, on=["period", "accepted_viewer"], how="left")
    grouped[["control_count", "control_proportion"]] = grouped[[
        "control_count", "control_proportion"
    ]].fillna(0)
    grouped["delta_count_vs_control"] = grouped["count"] - grouped["control_count"]
    grouped["delta_proportion_vs_control"] = grouped["proportion"] - grouped["control_proportion"]
    return grouped


def period_effects(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control = {
        item["prediction_period"]: item
        for item in runs
        if item["strategy_id"] == "expanding_history"
    }
    effects: list[dict[str, Any]] = []
    for item in runs:
        reference = control[item["prediction_period"]]
        effects.append({
            "strategy_id": item["strategy_id"],
            "prediction_period": item["prediction_period"],
            "delta_recall": float(item["metrics"]["recall"] - reference["metrics"]["recall"]),
            "delta_f1": float(item["metrics"]["f1"] - reference["metrics"]["f1"]),
            "delta_false_positives": int(item["metrics"]["fp"] - reference["metrics"]["fp"]),
            "p4_performance_preserved": (
                bool(all(item["metrics"][name] == reference["metrics"][name] for name in ("precision", "recall", "f1", "tn", "fp", "fn", "tp")))
                if item["prediction_period"] == "P4" else None
            ),
        })
    return effects


def _metric_fingerprint(runs: list[dict[str, Any]]) -> str:
    stable = [
        {
            "strategy_id": item["strategy_id"],
            "prediction_period": item["prediction_period"],
            "metrics": item["metrics"],
            "routing_counts": item["routing_counts"],
        }
        for item in runs
    ]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_phase8r1a(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    set_random_seed(int(config["seed"]))
    frozen_hashes = verify_frozen_chain(config)
    assigned, definitions = load_p0_p4_membership(config)
    phase7a_config = yaml.safe_load(config["phase7a_config"].read_text(encoding="utf-8"))
    resolved_frozen_config = {
        **phase7a_config,
        "paper_feature_config": config["paper_feature_config"],
        "paper_model_config": config["paper_model_config"],
        "paper_mulcas_config": config["paper_mulcas_config"],
    }
    feature_config, model_config, mulcas_config = _load_frozen_configs(resolved_frozen_config)

    strategies = config["strategies"]
    control = next(item for item in strategies if item["id"] == "expanding_history")
    runs: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []

    for period in PREDICTION_PERIODS:
        result, predictions, traces = evaluate_strategy_period(
            assigned, control, period, feature_config, model_config, mulcas_config, seed=int(config["seed"])
        )
        runs.append(result)
        prediction_frames.append(predictions)
        trace_frames.append(traces)

    control_reproduction = verify_control_reproduction(
        runs,
        config["phase7a_results"],
        float(config["control"]["tolerance"]),
    )
    if not control_reproduction["passed"]:
        raise RuntimeError("Expanding-history control did not reproduce Phase 7A; candidates were not run.")

    for strategy in strategies:
        if strategy["id"] == "expanding_history":
            continue
        for period in PREDICTION_PERIODS:
            result, predictions, traces = evaluate_strategy_period(
                assigned, strategy, period, feature_config, model_config, mulcas_config, seed=int(config["seed"])
            )
            runs.append(result)
            prediction_frames.append(predictions)
            trace_frames.append(traces)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    traces = pd.concat(trace_frames, ignore_index=True)
    if "P5" in set(predictions["period"]):
        raise AssertionError("P5 predictions were generated.")
    strategy_order = [item["id"] for item in strategies]
    summaries = summarize_strategies(
        runs, predictions, strategy_order, float(config["selection"]["precision_floor"])
    )
    selection = select_strategy(summaries)
    control_summary = next(item for item in summaries if item["strategy_id"] == "expanding_history")
    selected_id = selection["selected_strategy"] or selection["diagnostic_fallback"]
    selected_summary = next(item for item in summaries if item["strategy_id"] == selected_id)
    h8r1_supported = bool(
        selection["selected_strategy"]
        and selection["selected_strategy"] != "expanding_history"
        and selected_summary["macro_mean_f1"] > control_summary["macro_mean_f1"]
    )

    membership = load_subgroup_membership(config, definitions)
    subgroup = subgroup_metrics(predictions, membership)
    routing = routing_summary(predictions)
    run_table = pd.DataFrame([_flatten_run(item) for item in runs])
    summary_table = pd.DataFrame(summaries)
    period_table = run_table.copy()
    results = {
        "phase": "8R1A",
        "research_question": "RQ6",
        "hypothesis": config["hypothesis"],
        "h8r1_supported": h8r1_supported,
        "p5_loaded_or_evaluated": False,
        "random_seed": int(config["seed"]),
        "dataset_sha256": sha256_file(config["dataset"]),
        "config_sha256": config["config_sha256"],
        "frozen_artifact_hashes": frozen_hashes,
        "frozen_behavior": config["frozen_behavior"],
        "period_definitions": definitions,
        "control_reproduction": control_reproduction,
        "runs": runs,
        "strategy_summaries": summaries,
        "selection": selection,
        "period_specific_effects": period_effects(runs),
        "deterministic_metric_fingerprint": _metric_fingerprint(runs),
    }
    return results, run_table, summary_table, period_table, subgroup, routing, predictions, traces


def _format_metric(value: Any) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.4f}"


def build_report(results: dict[str, Any], subgroup: pd.DataFrame) -> str:
    run_rows = [
        "| Strategy | Period | Precision | Recall | F1 | PR-AUC | Balanced acc. | TN | FP | FN | TP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in results["runs"]:
        metric = run["metrics"]
        run_rows.append(
            f"| {run['strategy_id']} | {run['prediction_period']} | {metric['precision']:.4f} | "
            f"{metric['recall']:.4f} | {metric['f1']:.4f} | {metric['pr_auc']:.4f} | "
            f"{metric['balanced_accuracy']:.4f} | {metric['tn']} | {metric['fp']} | {metric['fn']} | {metric['tp']} |"
        )
    summary_rows = [
        "| Strategy | Mean precision | Mean recall | Mean F1 | F1 std | Mean PR-AUC | Mean balanced acc. | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results["strategy_summaries"]:
        summary_rows.append(
            f"| {item['strategy_id']} | {item['macro_mean_precision']:.4f} | {item['macro_mean_recall']:.4f} | "
            f"{item['macro_mean_f1']:.4f} | {item['macro_std_f1']:.4f} | {item['macro_mean_pr_auc']:.4f} | "
            f"{item['macro_mean_balanced_accuracy']:.4f} | {item['eligible']} |"
        )
    effect_rows = [
        "| Strategy | Period | Delta recall | Delta F1 | Delta FP | P4 preserved |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in results["period_specific_effects"]:
        effect_rows.append(
            f"| {item['strategy_id']} | {item['prediction_period']} | {item['delta_recall']:+.4f} | "
            f"{item['delta_f1']:+.4f} | {item['delta_false_positives']:+d} | "
            f"{_format_metric(item['p4_performance_preserved']) if item['p4_performance_preserved'] is None else item['p4_performance_preserved']} |"
        )
    subgroup_rows = [
        "| Strategy | Period | Subgroup | Positives | Recall |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    selected = results["selection"]["selected_strategy"] or results["selection"]["diagnostic_fallback"]
    for row in subgroup[subgroup["strategy_id"].isin(["expanding_history", selected])].itertuples(index=False):
        subgroup_rows.append(
            f"| {row.strategy_id} | {row.period} | {row.subgroup} | {row.positive_count} | {_format_metric(row.recall)} |"
        )
    status = "supported" if results["h8r1_supported"] else "not supported"
    return (
        "# Phase 8R1A: Recency-Aware Temporal Backtesting\n\n"
        "The expanding-history control reproduced frozen Phase 7A P2-P4 exactly. P5 was neither loaded nor evaluated.\n\n"
        "## Full MulCas Period Metrics\n\n" + "\n".join(run_rows) +
        "\n\n## Macro P2-P4 Strategy Summary\n\n" + "\n".join(summary_rows) +
        f"\n\nSelection status: `{results['selection']['selection_status']}`. "
        f"Selected strategy: `{results['selection']['selected_strategy']}`. "
        f"Diagnostic fallback: `{results['selection']['diagnostic_fallback']}`.\n\n"
        "## Period-Specific Effects vs Control\n\n" + "\n".join(effect_rows) +
        "\n\n## Novelty and Creator Diagnostics\n\n" + "\n".join(subgroup_rows) +
        f"\n\n## Conclusion\n\nH8R1 is **{status}** under the predefined macro-precision eligibility rule.\n"
    )


def save_outputs(
    results: dict[str, Any],
    runs: pd.DataFrame,
    summaries: pd.DataFrame,
    periods: pd.DataFrame,
    subgroup: pd.DataFrame,
    routing: pd.DataFrame,
    predictions: pd.DataFrame,
    traces: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "phase8r1a_results.json",
        "runs": output_dir / "phase8r1a_runs.csv",
        "summary": output_dir / "phase8r1a_strategy_summary.csv",
        "periods": output_dir / "phase8r1a_period_metrics.csv",
        "subgroups": output_dir / "phase8r1a_subgroup_metrics.csv",
        "routing": output_dir / "phase8r1a_routing.csv",
        "report": output_dir / "phase8r1a_report.md",
        "predictions": output_dir / "phase8r1a_predictions.csv",
        "traces": output_dir / "phase8r1a_traces.csv",
    }
    paths["results"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    runs.to_csv(paths["runs"], index=False)
    summaries.to_csv(paths["summary"], index=False)
    periods.to_csv(paths["periods"], index=False)
    subgroup.to_csv(paths["subgroups"], index=False)
    routing.to_csv(paths["routing"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    traces.to_csv(paths["traces"], index=False)
    paths["report"].write_text(build_report(results, subgroup), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = run_phase8r1a(config)
    for name, path in save_outputs(*outputs, config["output_dir"]).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
