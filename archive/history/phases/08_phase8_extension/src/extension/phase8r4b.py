"""Forward temporal classification with frozen Phase 8R4A creation features."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/extension/phase8r4b.yaml"
PREDICTION_PERIODS = ("P2", "P3", "P4")
EXPECTED_ROWS = {"P0": 104, "P1": 305, "P2": 5141, "P3": 289, "P4": 150}
VIEW_ARTIFACT_NAMES = {
    "creation_full": "creation_full",
    "deployment_init": "deployment_init",
    "combined": "creation_full_plus_init_and_boundary",
}
METRIC_NAMES = ("precision", "recall", "f1", "pr_auc", "balanced_accuracy")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in config["inputs"].items():
        if key != "expected_matrix_sha256":
            config["inputs"][key] = _resolve(value)
    config["frozen_manifests"] = [_resolve(value) for value in config["frozen_manifests"]]
    config["output_dir"] = _resolve(config["output_dir"])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["prediction_periods"] != list(PREDICTION_PERIODS) or config["prohibited_period"] != "P5":
        raise ValueError("Phase 8R4B must evaluate P2-P4 and prohibit P5.")
    if config["feature_views"] != {"creation_full": 85, "deployment_init": 90, "combined": 183}:
        raise ValueError("The frozen Phase 8R4A view dimensions changed.")
    logistic = config["models"]["logistic_regression"]
    expected_logistic = {
        "penalty": "l2", "C": 1.0, "class_weight": None,
        "solver": "liblinear", "max_iter": 2000, "threshold": 0.5,
        "scaler": "StandardScaler",
    }
    if logistic != expected_logistic:
        raise ValueError("The fixed Logistic Regression configuration changed.")
    xgboost = config["models"]["xgboost"]
    if float(xgboost["scale_pos_weight"]) != 1.0 or float(xgboost["threshold"]) != 0.5:
        raise ValueError("XGBoost weighting and threshold must remain neutral/default.")
    if config["selection"]["aggregation"] != "macro_period_mean":
        raise ValueError("Selection must use macro period means.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in config["frozen_manifests"]:
        observed.update(verify_frozen_manifest(manifest))
    matrix_hash = sha256_file(config["inputs"]["creation_feature_matrix"])
    if matrix_hash != config["inputs"]["expected_matrix_sha256"]:
        raise RuntimeError("Frozen Phase 8R4A feature matrix hash mismatch.")
    return observed


def load_p0_p4_metadata(config: dict[str, Any]) -> pd.DataFrame:
    """Read the exact frozen 5,989-row P0-P4 prefix, never P5."""
    definitions = json.loads(config["inputs"]["period_definitions"].read_text(encoding="utf-8"))
    selected = [item for item in definitions if item["period"] != "P5"]
    if tuple(item["period"] for item in selected) != ("P0", "P1", "P2", "P3", "P4"):
        raise ValueError("Frozen P0-P4 definitions are incomplete.")
    row_limit = sum(int(item["row_count"]) for item in selected)
    frame = pd.read_csv(
        config["inputs"]["dataset"],
        usecols=["address", "creator", "createdBlockNumber", "label"],
        nrows=row_limit,
        low_memory=False,
    ).reset_index(names="row_index")
    if len(frame) != 5989:
        raise AssertionError("P0-P4 loader must read exactly 5,989 rows.")
    frame["period"] = ""
    for definition in selected:
        mask = frame["createdBlockNumber"].between(
            int(definition["block_min"]), int(definition["block_max"]), inclusive="both"
        )
        frame.loc[mask, "period"] = definition["period"]
        rows = frame[mask]
        if len(rows) != int(definition["row_count"]) or int(rows["label"].sum()) != int(definition["positive_count"]):
            raise AssertionError(f"Frozen membership mismatch for {definition['period']}.")
    if frame.groupby("period").size().to_dict() != EXPECTED_ROWS or "P5" in set(frame["period"]):
        raise AssertionError("P5 entered the Phase 8R4B temporal frame.")
    return frame.sort_values(["createdBlockNumber", "row_index"], kind="mergesort").reset_index(drop=True)


def load_frozen_views(
    metadata: pd.DataFrame, config: dict[str, Any]
) -> dict[str, np.ndarray]:
    schema = json.loads(config["inputs"]["creation_feature_schema"].read_text(encoding="utf-8"))
    views: dict[str, np.ndarray] = {}
    with np.load(config["inputs"]["creation_feature_matrix"], allow_pickle=False) as artifact:
        row_index = artifact["row_index"]
        if not np.array_equal(row_index, np.arange(6498, dtype=np.int64)):
            raise AssertionError("Frozen matrix row identity changed.")
        positions = metadata["row_index"].to_numpy(dtype=int)
        for view, dimension in config["feature_views"].items():
            artifact_name = VIEW_ARTIFACT_NAMES[view]
            matrix = np.asarray(artifact[artifact_name][positions], dtype=np.float32)
            schema_view = "creation_full_plus_init_and_boundary" if view == "combined" else view
            if matrix.shape != (len(metadata), int(dimension)):
                raise AssertionError(f"Frozen matrix shape changed for {view}.")
            if int(schema["views"][schema_view]["dimension"]) != int(dimension):
                raise AssertionError(f"Frozen schema dimension changed for {view}.")
            if not np.isfinite(matrix).all():
                raise AssertionError(f"Non-finite frozen features in {view}.")
            views[view] = matrix
    return views


def load_diagnostics(metadata: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    target_count = sum(EXPECTED_ROWS[period] for period in PREDICTION_PERIODS)
    clone = pd.read_csv(
        config["inputs"]["phase7_clone_membership"],
        usecols=[
            "row_index", "period", "previously_seen_creator",
            "has_any_exact_historical_clone", "creator_history_feature",
        ],
        nrows=target_count,
    )
    mulcas = pd.read_csv(
        config["inputs"]["phase7_predictions"],
        usecols=["row_index", "period", "label", "final_prediction", "final_score"],
        nrows=target_count,
    ).rename(columns={"final_prediction": "mulcas_prediction", "final_score": "mulcas_score"})
    creation = pd.read_csv(
        config["inputs"]["creation_feature_summary"],
        usecols=["row_index", "period", "init_region_available"],
        nrows=len(metadata),
    )
    creation = creation[creation["period"].isin(PREDICTION_PERIODS)]
    diagnostics = mulcas.merge(clone, on=["row_index", "period"], validate="one_to_one")
    diagnostics = diagnostics.merge(creation, on=["row_index", "period"], validate="one_to_one")
    expected = metadata[metadata["period"].isin(PREDICTION_PERIODS)]["row_index"]
    if set(diagnostics["row_index"]) != set(expected) or set(diagnostics["period"]) != set(PREDICTION_PERIODS):
        raise AssertionError("Frozen diagnostic membership does not match P2-P4.")
    if "P5" in set(diagnostics["period"]):
        raise AssertionError("P5 diagnostic rows were loaded.")
    return diagnostics


def _make_model(model_name: str, config: dict[str, Any]) -> Any:
    seed = int(config["random_seed"])
    if model_name == "logistic_regression":
        values = config["models"][model_name]
        return LogisticRegression(
            l1_ratio=0, C=float(values["C"]), class_weight=None,
            solver=values["solver"], max_iter=int(values["max_iter"]),
            random_state=seed,
        )
    if model_name == "xgboost":
        values = {key: value for key, value in config["models"][model_name].items() if key != "threshold"}
        return XGBClassifier(random_state=seed, **values)
    raise ValueError(f"Unknown fixed model: {model_name}")


def fit_predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_target: np.ndarray,
    train_row_indices: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler_fit_rows: list[int] = []
    if model_name == "logistic_regression":
        scaler = StandardScaler().fit(x_train)
        x_train_used = scaler.transform(x_train)
        x_target_used = scaler.transform(x_target)
        scaler_fit_rows = train_row_indices.astype(int).tolist()
    else:
        x_train_used, x_target_used = x_train, x_target
    model = _make_model(model_name, config)
    started = time.perf_counter()
    model.fit(x_train_used, y_train)
    probability = np.asarray(model.predict_proba(x_target_used))[:, 1]
    elapsed = time.perf_counter() - started
    threshold = float(config["models"][model_name]["threshold"])
    prediction = (probability >= threshold).astype(int)
    diagnostics = {
        "runtime_seconds": elapsed,
        "threshold": threshold,
        "scaler_fit_row_indices": scaler_fit_rows,
        "class_weight": None,
        "sampling": "none",
        "iterations": (
            [int(value) for value in model.n_iter_]
            if model_name == "logistic_regression" else None
        ),
    }
    return probability, prediction, diagnostics


def performance_preflight(
    metadata: pd.DataFrame, views: dict[str, np.ndarray], config: dict[str, Any]
) -> dict[str, Any]:
    train_mask = metadata["period"].isin(config["preflight"]["train_periods"]).to_numpy()
    target_mask = metadata["period"].eq(config["preflight"]["prediction_period"]).to_numpy()
    matrix = views[config["preflight"]["feature_view"]]
    runtimes: dict[str, float] = {}
    for model_name in ("logistic_regression", "xgboost"):
        started = time.perf_counter()
        fit_predict(
            model_name, matrix[train_mask], metadata.loc[train_mask, "label"].to_numpy(dtype=int),
            matrix[target_mask], metadata.loc[train_mask, "row_index"].to_numpy(dtype=int), config,
        )
        runtimes[model_name] = time.perf_counter() - started
    estimate = sum(runtimes.values()) * 9
    return {
        "cpu_model": platform.processor(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "system_ram_bytes": int(psutil.virtual_memory().total),
        "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
        "execution_device": "cpu", "representative_train_periods": config["preflight"]["train_periods"],
        "representative_prediction_period": config["preflight"]["prediction_period"],
        "representative_feature_view": config["preflight"]["feature_view"],
        "representative_runtime_seconds": runtimes,
        "estimated_full_runtime_seconds": estimate,
        "stop_limit_seconds": float(config["preflight"]["maximum_estimated_runtime_seconds"]),
        "full_experiment_authorized": estimate <= float(config["preflight"]["maximum_estimated_runtime_seconds"]),
    }


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return binary_classification_metrics(
        frame["label"].to_numpy(dtype=int), prediction, probability
    )


def frozen_mulcas_metrics(diagnostics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for period in PREDICTION_PERIODS:
        selected = diagnostics[diagnostics["period"].eq(period)]
        results[period] = _metrics(
            selected, selected["mulcas_prediction"].to_numpy(dtype=int),
            selected["mulcas_score"].to_numpy(dtype=float),
        )
    return results


def summarize_candidates(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    simplicity = {name: index for index, name in enumerate(config["selection"]["simplicity_order"])}
    for (model, view), group in metrics.groupby(["model", "feature_view"], sort=False):
        candidate = f"{model}__{view}"
        row: dict[str, Any] = {
            "candidate_id": candidate, "model": model, "feature_view": view,
            "feature_dimension": int(config["feature_views"][view]),
            "simplicity_rank": simplicity[candidate],
        }
        for metric in METRIC_NAMES:
            row[f"macro_mean_{metric}"] = float(group[metric].mean())
            row[f"macro_std_{metric}"] = float(group[metric].std(ddof=0))
        row["eligible"] = row["macro_mean_precision"] >= float(config["selection"]["precision_floor"])
        rows.append(row)
    return pd.DataFrame(rows)


def select_candidate(summary: pd.DataFrame) -> dict[str, Any]:
    eligible = summary[summary["eligible"].astype(bool)]
    pool = eligible if len(eligible) else summary
    ranked = pool.sort_values(
        ["macro_mean_f1", "macro_mean_recall", "macro_mean_balanced_accuracy", "simplicity_rank"],
        ascending=[False, False, False, True], kind="mergesort",
    )
    selected = ranked.iloc[0]
    return {
        "candidate_id": str(selected["candidate_id"]),
        "model": str(selected["model"]), "feature_view": str(selected["feature_view"]),
        "eligible": bool(selected["eligible"]),
        "selection_reason": (
            "Highest-ranked eligible macro P2-P4 candidate."
            if bool(selected["eligible"]) else
            "No candidate met macro precision >= 0.80; selected as diagnostic fallback."
        ),
    }


def subgroup_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positive_groups = {
        "historically_novel_ponzi": lambda x: x["label"].eq(1) & ~x["has_any_exact_historical_clone"].astype(bool),
        "exact_clone_ponzi": lambda x: x["label"].eq(1) & x["has_any_exact_historical_clone"].astype(bool),
        "unseen_creator_ponzi": lambda x: x["label"].eq(1) & ~x["previously_seen_creator"].astype(bool),
        "seen_creator_ponzi": lambda x: x["label"].eq(1) & x["previously_seen_creator"].astype(bool),
    }
    for (candidate, period), frame in predictions.groupby(["candidate_id", "period"], sort=True):
        for name, selector in positive_groups.items():
            selected = frame[selector(frame)]
            support = len(selected)
            tp = int(selected["prediction"].eq(1).sum())
            rows.append({
                "candidate_id": candidate, "period": period, "group_type": "positive_recall",
                "group": name, "row_count": support, "positive_count": support,
                "recall": tp / support if support else None,
                "precision": None, "f1": None, "balanced_accuracy": None,
                "tn": None, "fp": None, "fn": support - tp, "tp": tp,
                "selection_use": "diagnostic_only",
            })
        for available in (1, 0):
            selected = frame[frame["init_region_available"].astype(int).eq(available)]
            if selected.empty:
                values = {name: None for name in ("precision", "recall", "f1", "balanced_accuracy", "tn", "fp", "fn", "tp")}
            else:
                measured = _metrics(
                    selected, selected["prediction"].to_numpy(dtype=int),
                    selected["probability"].to_numpy(dtype=float),
                )
                values = {name: measured[name] for name in ("precision", "recall", "f1", "balanced_accuracy", "tn", "fp", "fn", "tp")}
            rows.append({
                "candidate_id": candidate, "period": period, "group_type": "init_availability",
                "group": f"init_region_available_{available}", "row_count": len(selected),
                "positive_count": int(selected["label"].sum()) if len(selected) else 0,
                **values, "selection_use": "diagnostic_only",
            })
    return pd.DataFrame(rows)


def complementarity(predictions: pd.DataFrame, selected_id: str) -> pd.DataFrame:
    selected = predictions[predictions["candidate_id"].eq(selected_id)]
    rows: list[dict[str, Any]] = []
    for period in (*PREDICTION_PERIODS, "P2-P4"):
        frame = selected if period == "P2-P4" else selected[selected["period"].eq(period)]
        truth = frame["label"].astype(int)
        mulcas = frame["mulcas_prediction"].astype(int)
        creation = frame["prediction"].astype(int)
        mulcas_correct, creation_correct = mulcas.eq(truth), creation.eq(truth)
        positive = truth.eq(1)
        m_tp, c_tp = mulcas.eq(1) & positive, creation.eq(1) & positive
        support = int(positive.sum())
        rows.append({
            "candidate_id": selected_id, "period": period, "row_count": len(frame),
            "both_correct": int((mulcas_correct & creation_correct).sum()),
            "both_wrong": int((~mulcas_correct & ~creation_correct).sum()),
            "mulcas_only_correct": int((mulcas_correct & ~creation_correct).sum()),
            "creation_only_correct": int((~mulcas_correct & creation_correct).sum()),
            "positive_support": support,
            "tp_by_both": int((m_tp & c_tp).sum()),
            "mulcas_only_tp": int((m_tp & ~c_tp).sum()),
            "creation_only_tp": int((~m_tp & c_tp).sum()),
            "fn_by_both": int((~m_tp & ~c_tp & positive).sum()),
            "mulcas_recall": float(m_tp.sum() / support),
            "creation_recall": float(c_tp.sum() / support),
            "oracle_union_recall": float((m_tp | c_tp).sum() / support),
            "oracle_non_deployable": True,
        })
    return pd.DataFrame(rows)


def build_report(
    summary: pd.DataFrame, metrics: pd.DataFrame, selection: dict[str, Any],
    complement: pd.DataFrame, preflight: dict[str, Any], actual_runtime: float,
    supported: bool,
) -> str:
    summary_rows = [
        f"| {row.candidate_id} | {row.macro_mean_precision:.4f} | {row.macro_mean_recall:.4f} | "
        f"{row.macro_mean_f1:.4f} | {row.macro_mean_pr_auc:.4f} | "
        f"{row.macro_mean_balanced_accuracy:.4f} | {bool(row.eligible)} |"
        for row in summary.itertuples(index=False)
    ]
    selected_metrics = metrics[metrics["candidate_id"].eq(selection["candidate_id"])]
    period_rows = [
        f"| {row.period} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | "
        f"{row.pr_auc:.4f} | {row.balanced_accuracy:.4f} | "
        f"{int(row.tn)}/{int(row.fp)}/{int(row.fn)}/{int(row.tp)} | {row.delta_f1_vs_mulcas:+.4f} |"
        for row in selected_metrics.itertuples(index=False)
    ]
    oracle_rows = [
        f"| {row.period} | {int(row.tp_by_both)} | {int(row.mulcas_only_tp)} | "
        f"{int(row.creation_only_tp)} | {int(row.fn_by_both)} | {row.oracle_union_recall:.4f} |"
        for row in complement.itertuples(index=False)
    ]
    return "\n".join([
        "# Phase 8R4B: Forward Creation-Code Classification", "",
        "The frozen Phase 8R4A matrices were evaluated with six predeclared candidates on P2-P4 only. P5 was never loaded, no threshold was tuned, and MulCas was not retrained.", "",
        "## Runtime Preflight", "",
        f"CPU execution was used (`torch {preflight['torch_version']}`, CUDA={preflight['cuda_available']}). Representative runtimes were {preflight['representative_runtime_seconds']}; projected full runtime was {preflight['estimated_full_runtime_seconds']:.2f}s and actual runtime was {actual_runtime:.2f}s.", "",
        "## Candidate Summary", "", "| Candidate | Macro precision | Macro recall | Macro F1 | Macro PR-AUC | Macro BA | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |", *summary_rows, "",
        f"Selected: **{selection['candidate_id']}**. Eligible: **{selection['eligible']}**. {selection['selection_reason']}", "",
        "## Selected Candidate By Period", "", "| Period | Precision | Recall | F1 | PR-AUC | BA | TN/FP/FN/TP | Delta F1 vs MulCas |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |", *period_rows, "",
        "## Positive Complementarity", "", "| Period | TP both | MulCas-only TP | Creation-only TP | FN both | Oracle union recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |", *oracle_rows, "",
        "Oracle union recall is a non-deployable diagnostic; no hybrid was trained.", "",
        "## Conclusion", "",
        f"H8R4 is **{'supported' if supported else 'not supported'}** under the predefined macro-precision eligibility rule.", "",
    ])


def run(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    set_random_seed(int(config["random_seed"]))
    frozen = verify_frozen_inputs(config)
    metadata = load_p0_p4_metadata(config)
    views = load_frozen_views(metadata, config)
    diagnostics = load_diagnostics(metadata, config)
    mulcas = frozen_mulcas_metrics(diagnostics)
    preflight = performance_preflight(metadata, views, config)
    if not preflight["full_experiment_authorized"]:
        config["output_dir"].mkdir(parents=True, exist_ok=True)
        (config["output_dir"] / "runtime_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        raise RuntimeError("Estimated Phase 8R4B runtime exceeds 30 minutes.")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fit_diagnostics: dict[str, Any] = {}
    position = pd.Series(np.arange(len(metadata)), index=metadata["row_index"])
    for model_name in ("logistic_regression", "xgboost"):
        for view, matrix in views.items():
            candidate = f"{model_name}__{view}"
            for period in PREDICTION_PERIODS:
                target_number = int(period[1:])
                train_periods = [f"P{index}" for index in range(target_number)]
                train = metadata[metadata["period"].isin(train_periods)]
                target = metadata[metadata["period"].eq(period)]
                train_positions = position.loc[train["row_index"]].to_numpy(dtype=int)
                target_positions = position.loc[target["row_index"]].to_numpy(dtype=int)
                probability, prediction, detail = fit_predict(
                    model_name, matrix[train_positions], train["label"].to_numpy(dtype=int),
                    matrix[target_positions], target["row_index"].to_numpy(dtype=int), config,
                )
                measured = _metrics(target, prediction, probability)
                control = mulcas[period]
                metric_row = {
                    "candidate_id": candidate, "model": model_name, "feature_view": view,
                    "feature_dimension": int(config["feature_views"][view]),
                    "period": period, "train_periods": "+".join(train_periods),
                    "train_row_count": len(train), "prediction_row_count": len(target),
                    "runtime_seconds": detail["runtime_seconds"], **measured,
                }
                for metric in METRIC_NAMES:
                    metric_row[f"delta_{metric}_vs_mulcas"] = measured[metric] - control[metric]
                metric_row.update({
                    "delta_tn_vs_mulcas": measured["tn"] - control["tn"],
                    "delta_fp_vs_mulcas": measured["fp"] - control["fp"],
                    "delta_fn_vs_mulcas": measured["fn"] - control["fn"],
                    "delta_tp_vs_mulcas": measured["tp"] - control["tp"],
                })
                metric_rows.append(metric_row)
                fit_diagnostics[f"{candidate}__{period}"] = detail
                exported = target[["row_index", "address", "period", "createdBlockNumber", "label"]].copy()
                exported.insert(0, "candidate_id", candidate)
                exported["model"] = model_name
                exported["feature_view"] = view
                exported["probability"] = probability
                exported["prediction"] = prediction
                exported = exported.merge(diagnostics, on=["row_index", "period", "label"], validate="one_to_one")
                prediction_frames.append(exported)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_candidates(metrics, config)
    selection = select_candidate(summary)
    summary["selection_status"] = "not_selected"
    summary.loc[summary["candidate_id"].eq(selection["candidate_id"]), "selection_status"] = (
        "selected" if selection["eligible"] else "diagnostic_fallback"
    )
    subgroups = subgroup_diagnostics(predictions)
    complement = complementarity(predictions, selection["candidate_id"])
    supported = bool(selection["eligible"])
    actual_runtime = time.perf_counter() - started
    preflight["actual_full_runtime_seconds"] = actual_runtime
    results = {
        "phase": "8R4B", "hypothesis": config["hypothesis"],
        "p5_loaded_or_evaluated": False, "mulcas_retrained": False,
        "feature_selection_performed": False, "threshold_tuned": False,
        "class_weighting": "none", "sampling": "none",
        "native_threshold": 0.5, "random_seed": int(config["random_seed"]),
        "matrix_sha256": sha256_file(config["inputs"]["creation_feature_matrix"]),
        "schema_sha256": sha256_file(config["inputs"]["creation_feature_schema"]),
        "frozen_artifact_hashes": frozen, "config_sha256": config["config_sha256"],
        "preflight": preflight, "fixed_models": config["models"],
        "selection": selection, "h8r4_supported": supported,
        "fit_diagnostics": fit_diagnostics,
    }
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "model_period_metrics.csv", index=False)
    summary.to_csv(output / "candidate_summary.csv", index=False)
    subgroups.to_csv(output / "subgroup_metrics.csv", index=False)
    predictions.to_csv(output / "creation_predictions.csv", index=False)
    complement.to_csv(output / "complementarity.csv", index=False)
    (output / "runtime_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    (output / "phase8r4b_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "phase8r4b_report.md").write_text(
        build_report(summary, metrics, selection, complement, preflight, actual_runtime, supported),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    result = run(load_config(args.config))
    print(json.dumps({
        "selected_candidate": result["selection"]["candidate_id"],
        "eligible": result["selection"]["eligible"],
        "h8r4_supported": result["h8r4_supported"],
    }, indent=2))


if __name__ == "__main__":
    main()
