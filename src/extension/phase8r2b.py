"""Run Phase 8R2B graph aggregate and temporal GIN backtests."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch_geometric
import yaml

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.extension.graph_dataset import (
    EXPECTED_CORPUS_SHA256,
    EXPECTED_SCHEMA_SHA256,
    FrozenGraphStore,
    aggregate_feature_names,
    chronological_inner_split,
    fit_node_normalizer,
    graph_aggregate_vector,
    normalized_graphs,
)
from src.extension.graph_models import (
    GINClassifier,
    graph_loader,
    predict_probabilities,
    set_torch_seed,
    train_with_early_stopping,
)
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/extension/phase8r2b.yaml"
METRICS = ("precision", "recall", "f1", "pr_auc", "roc_auc", "balanced_accuracy", "accuracy")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset", "graph_manifest", "graph_schema", "graph_statistics", "period_definitions",
        "phase7a_results", "phase7a_predictions", "phase7e_membership", "frozen_phase7_manifest",
        "frozen_phase8r0_manifest", "frozen_phase8r1a_manifest", "frozen_phase8r2a_manifest", "output_dir",
    ):
        config[key] = _resolve(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    architectures = list(product(config["gin"]["layers"], config["gin"]["hidden_dimensions"], config["gin"]["dropouts"]))
    if len(architectures) != 8 or config["gin"]["seeds"] != [7, 42, 123, 2026, 999]:
        raise ValueError("Phase 8R2B requires exactly eight architectures and five frozen seeds.")
    if config["periods"]["prohibited"] != "P5" or any(
        scenario["predict"] == "P5" or "P5" in scenario["train"] for scenario in config["periods"]["scenarios"]
    ):
        raise ValueError("P5 is prohibited from Phase 8R2B.")
    if float(config["gin"]["threshold"]) != 0.5 or config["aggregate_baseline"]["class_weight"] is not None:
        raise ValueError("Threshold and unweighted aggregate baseline are frozen.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in (
        config["frozen_phase7_manifest"], config["frozen_phase8r0_manifest"],
        config["frozen_phase8r1a_manifest"], config["frozen_phase8r2a_manifest"],
    ):
        observed.update(verify_frozen_manifest(manifest))
    graph_manifest = json.loads(config["graph_manifest"].read_text(encoding="utf-8"))
    if graph_manifest["corpus_sha256"] != EXPECTED_CORPUS_SHA256:
        raise RuntimeError("Phase 8R2A corpus hash changed.")
    if graph_manifest["schema_sha256"] != EXPECTED_SCHEMA_SHA256:
        raise RuntimeError("Phase 8R2A schema hash changed.")
    return observed


def load_temporal_frame(config: dict[str, Any]) -> pd.DataFrame:
    definitions = json.loads(config["period_definitions"].read_text(encoding="utf-8"))
    allowed = [item for item in definitions if item["period"] != "P5"]
    nrows = sum(int(item["row_count"]) for item in allowed)
    frame = pd.read_csv(
        config["dataset"], usecols=["address", "createdBlockNumber", "label"], nrows=nrows, low_memory=False
    ).reset_index(names="row_index")
    boundaries = [(item["period"], int(item["block_min"]), int(item["block_max"])) for item in allowed]
    frame["period"] = frame["createdBlockNumber"].map(
        lambda value: next(period for period, minimum, maximum in boundaries if minimum <= int(value) <= maximum)
    )
    if len(frame) != 5989 or set(frame["period"]) != {"P0", "P1", "P2", "P3", "P4"}:
        raise AssertionError("Exact frozen P0-P4 memberships were not loaded.")
    if int(frame["createdBlockNumber"].max()) >= int(next(x for x in definitions if x["period"] == "P5")["block_min"]):
        raise AssertionError("P5 rows entered Phase 8R2B.")
    frame["label"] = frame["label"].astype(int)
    return frame


def architecture_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"architecture_id": f"gin_l{layers}_h{hidden}_d{str(dropout).replace('.', '')}",
         "layers": int(layers), "hidden_dim": int(hidden), "dropout": float(dropout)}
        for layers, hidden, dropout in product(
            config["gin"]["layers"], config["gin"]["hidden_dimensions"], config["gin"]["dropouts"]
        )
    ]


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return binary_classification_metrics(labels, (scores >= 0.5).astype(int), scores)


def run_aggregate_baseline(
    config: dict[str, Any], frame: pd.DataFrame, graphs: dict[int, Any], statistics: pd.DataFrame, node_names: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    feature_names = aggregate_feature_names(node_names)
    stat_index = statistics.set_index("row_index")
    vectors = {
        row: graph_aggregate_vector(graphs[row], stat_index.loc[row]) for row in frame["row_index"].astype(int)
    }
    rows: list[dict[str, Any]] = []
    baseline = config["aggregate_baseline"]
    for scenario in config["periods"]["scenarios"]:
        train = frame[frame["period"].isin(scenario["train"])]
        evaluate = frame[frame["period"].eq(scenario["predict"])]
        x_train = np.vstack([vectors[int(row)] for row in train["row_index"]])
        x_evaluate = np.vstack([vectors[int(row)] for row in evaluate["row_index"]])
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(
            C=float(baseline["C"]), penalty=baseline["penalty"], solver=baseline["solver"],
            class_weight=None, max_iter=int(baseline["max_iter"]), random_state=42,
        )
        started = time.perf_counter()
        model.fit(scaler.transform(x_train), train["label"].to_numpy())
        scores = model.predict_proba(scaler.transform(x_evaluate))[:, 1]
        elapsed = time.perf_counter() - started
        rows.append({
            "model": "graph_aggregate_logistic_regression", "prediction_period": scenario["predict"],
            "training_periods": "+".join(scenario["train"]), "train_rows": len(train),
            "train_positive": int(train["label"].sum()), "train_negative": int((train["label"] == 0).sum()),
            "prediction_rows": len(evaluate), "feature_dimension": len(feature_names), "scaler_fit_rows": len(train),
            "threshold": 0.5, "runtime_seconds": elapsed, **_metrics(evaluate["label"].to_numpy(), scores),
        })
    return pd.DataFrame(rows), feature_names


def _scenario_data(frame: pd.DataFrame, scenario: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = frame[frame["period"].isin(scenario["train"])].copy()
    future = frame[frame["period"].eq(scenario["predict"])].copy()
    if historical["createdBlockNumber"].max() >= future["createdBlockNumber"].min():
        raise AssertionError("Future period does not strictly follow historical training.")
    return historical, future


def run_gin_experiments(
    config: dict[str, Any], frame: pd.DataFrame, graphs: dict[int, Any], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_path = output_dir / "gin_runs.csv"
    prediction_path = output_dir / "gin_predictions.csv"
    diagnostic_path = output_dir / "training_diagnostics.csv"
    existing_runs = pd.read_csv(run_path).to_dict("records") if run_path.exists() else []
    existing_predictions = pd.read_csv(prediction_path).to_dict("records") if prediction_path.exists() else []
    existing_diagnostics = pd.read_csv(diagnostic_path).to_dict("records") if diagnostic_path.exists() else []
    completed = {(row["architecture_id"], int(row["seed"]), row["prediction_period"]) for row in existing_runs}
    runs, predictions, diagnostics = list(existing_runs), list(existing_predictions), list(existing_diagnostics)
    labels = dict(zip(frame["row_index"].astype(int), frame["label"].astype(int), strict=True))
    gin = config["gin"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for scenario in config["periods"]["scenarios"]:
        historical, future = _scenario_data(frame, scenario)
        inner_train_rows, inner_valid_rows, split_policy = chronological_inner_split(
            historical, float(gin["inner_validation_fraction"])
        )
        train_raw = [graphs[int(row)] for row in inner_train_rows]
        valid_raw = [graphs[int(row)] for row in inner_valid_rows]
        future_raw = [graphs[int(row)] for row in future["row_index"]]
        normalizer = fit_node_normalizer(train_raw, inner_train_rows)
        train_graphs = normalized_graphs(train_raw, normalizer)
        valid_graphs = normalized_graphs(valid_raw, normalizer)
        future_graphs = normalized_graphs(future_raw, normalizer)
        inner_labels = np.asarray([labels[int(row)] for row in inner_train_rows])
        pos_weight = float((inner_labels == 0).sum() / (inner_labels == 1).sum())

        for architecture in architecture_grid(config):
            for seed in gin["seeds"]:
                key = (architecture["architecture_id"], int(seed), scenario["predict"])
                if key in completed:
                    continue
                set_torch_seed(int(seed), bool(gin["deterministic_algorithms"]))
                train_loader = graph_loader(train_graphs, int(gin["batch_size"]), True, int(seed))
                valid_loader = graph_loader(valid_graphs, int(gin["evaluation_batch_size"]), False, int(seed))
                future_loader = graph_loader(future_graphs, int(gin["evaluation_batch_size"]), False, int(seed))
                model = GINClassifier(55, architecture["hidden_dim"], architecture["layers"], architecture["dropout"])
                started = time.perf_counter()
                trained = train_with_early_stopping(
                    model, train_loader, valid_loader, pos_weight, float(gin["learning_rate"]),
                    float(gin["weight_decay"]), int(gin["max_epochs"]),
                    int(gin["early_stopping_patience"]), device,
                )
                row_order, scores = predict_probabilities(trained.model, future_loader, device)
                elapsed = time.perf_counter() - started
                expected_rows = future["row_index"].to_numpy(dtype=int)
                if not np.array_equal(row_order, expected_rows):
                    raise AssertionError("Prediction identity/order changed.")
                truth = future["label"].to_numpy(dtype=int)
                metrics = _metrics(truth, scores)
                run_id = f"{architecture['architecture_id']}__s{seed}__{scenario['predict']}"
                run = {
                    "run_id": run_id, **architecture, "seed": int(seed), "prediction_period": scenario["predict"],
                    "training_periods": "+".join(scenario["train"]), "inner_split_policy": split_policy,
                    "inner_train_rows": len(inner_train_rows), "inner_train_positive": int(inner_labels.sum()),
                    "inner_train_negative": int((inner_labels == 0).sum()), "inner_validation_rows": len(inner_valid_rows),
                    "inner_validation_positive": int(sum(labels[int(x)] for x in inner_valid_rows)),
                    "inner_validation_negative": int(sum(labels[int(x)] == 0 for x in inner_valid_rows)),
                    "prediction_rows": len(future), "pos_weight": pos_weight, "threshold": 0.5,
                    "device": str(device), "runtime_seconds": elapsed, "epochs_trained": trained.epochs_trained,
                    "best_inner_validation_epoch": trained.best_epoch, "training_loss": trained.training_loss,
                    "validation_loss": trained.validation_loss, **metrics,
                }
                runs.append(run)
                for position, row_index in enumerate(row_order):
                    predictions.append({
                        "run_id": run_id, "architecture_id": architecture["architecture_id"], "seed": int(seed),
                        "period": scenario["predict"], "row_index": int(row_index), "label": int(truth[position]),
                        "probability": float(scores[position]), "prediction": int(scores[position] >= 0.5),
                    })
                diagnostics.append({
                    "run_id": run_id, "prediction_period": scenario["predict"], "seed": int(seed),
                    "normalizer_fit_row_count": len(normalizer.fitted_row_indices),
                    "normalizer_fit_row_min": min(normalizer.fitted_row_indices),
                    "normalizer_fit_row_max": max(normalizer.fitted_row_indices),
                    "normalizer_mean": json.dumps(normalizer.mean.tolist(), separators=(",", ":")),
                    "normalizer_std": json.dumps(normalizer.std.tolist(), separators=(",", ":")),
                    "pos_weight_source": "inner_train_labels_only", "early_stopping_source": "inner_validation_only",
                    "future_used_for_early_stopping": False,
                })
                pd.DataFrame(runs).to_csv(run_path, index=False)
                pd.DataFrame(predictions).to_csv(prediction_path, index=False)
                pd.DataFrame(diagnostics).to_csv(diagnostic_path, index=False)
                print(f"completed {run_id}: F1={metrics['f1']:.4f}, epochs={trained.epochs_trained}, {elapsed:.1f}s", flush=True)
    return pd.DataFrame(runs), pd.DataFrame(predictions), pd.DataFrame(diagnostics), pd.DataFrame()


def summarize_gin(runs: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    period_rows: list[dict[str, Any]] = []
    for (architecture, period), group in runs.groupby(["architecture_id", "prediction_period"], sort=True):
        source = group.iloc[0]
        row: dict[str, Any] = {
            "architecture_id": architecture, "prediction_period": period, "layers": int(source["layers"]),
            "hidden_dim": int(source["hidden_dim"]), "dropout": float(source["dropout"]), "seed_count": len(group),
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        for metric in ("tn", "fp", "fn", "tp", "runtime_seconds", "epochs_trained"):
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        period_rows.append(row)
    period_frame = pd.DataFrame(period_rows)
    architecture_rows: list[dict[str, Any]] = []
    for architecture, group in period_frame.groupby("architecture_id", sort=True):
        source = group.iloc[0]
        original = runs[runs["architecture_id"].eq(architecture)]
        seed_macros = original.groupby("seed", sort=True)[list(METRICS)].mean()
        row = {
            "architecture_id": architecture, "layers": int(source["layers"]), "hidden_dim": int(source["hidden_dim"]),
            "dropout": float(source["dropout"]), "run_count": len(original),
        }
        for metric in METRICS:
            row[f"macro_mean_{metric}"] = float(group[f"mean_{metric}"].mean())
            row[f"macro_std_across_seeds_{metric}"] = float(seed_macros[metric].std(ddof=0))
        row["mean_runtime_seconds"] = float(original["runtime_seconds"].mean())
        row["total_runtime_seconds"] = float(original["runtime_seconds"].sum())
        row["eligible"] = row["macro_mean_precision"] >= float(config["selection"]["minimum_macro_precision"])
        architecture_rows.append(row)
    architecture_frame = pd.DataFrame(architecture_rows)
    eligible = architecture_frame[architecture_frame["eligible"]]
    pool = eligible if not eligible.empty else architecture_frame
    ranked = pool.sort_values(
        ["macro_mean_f1", "macro_mean_recall", "macro_mean_balanced_accuracy", "layers", "hidden_dim", "dropout"],
        ascending=[False, False, False, True, True, True], kind="mergesort",
    )
    selected = ranked.iloc[0].to_dict()
    selected["selection_status"] = "eligible" if not eligible.empty else "diagnostic_fallback_no_eligible_gin"
    selected["selection_reason"] = (
        "precision-eligible architecture ranked by macro F1, recall, balanced accuracy, then size"
        if not eligible.empty else "diagnostic fallback ranked identically after no architecture met macro precision >= 0.80"
    )
    return period_frame, architecture_frame, selected


def subgroup_metrics(predictions: pd.DataFrame, membership_path: Path) -> pd.DataFrame:
    count = sum((5141, 289, 150))
    membership = pd.read_csv(membership_path, nrows=count)
    membership = membership[membership["period"].isin(["P2", "P3", "P4"])][
        ["row_index", "previously_seen_creator", "has_any_exact_historical_clone"]
    ]
    merged = predictions.merge(membership, on="row_index", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    definitions = {
        "historically_novel_ponzi": lambda x: ~x["has_any_exact_historical_clone"].astype(bool),
        "cloned_ponzi": lambda x: x["has_any_exact_historical_clone"].astype(bool),
        "unseen_creator_ponzi": lambda x: ~x["previously_seen_creator"].astype(bool),
        "seen_creator_ponzi": lambda x: x["previously_seen_creator"].astype(bool),
    }
    for (architecture, seed, period), group in merged.groupby(["architecture_id", "seed", "period"], sort=True):
        positives = group[group["label"].eq(1)]
        for name, selector in definitions.items():
            selected = positives[selector(positives)]
            tp = int(selected["prediction"].sum())
            support = len(selected)
            rows.append({
                "architecture_id": architecture, "seed": int(seed), "period": period, "subgroup": name,
                "positive_support": support, "tp": tp, "fn": support - tp,
                "recall": float(tp / support) if support else None,
            })
    return pd.DataFrame(rows)


def frozen_mulcas_metrics(path: Path) -> pd.DataFrame:
    results = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scenario in results["scenarios"]:
        if scenario["prediction_period"] in ("P2", "P3", "P4"):
            rows.append({"prediction_period": scenario["prediction_period"], **scenario["metrics"]})
    return pd.DataFrame(rows)


def build_report(
    aggregate: pd.DataFrame, architectures: pd.DataFrame, periods: pd.DataFrame, selected: dict[str, Any],
    mulcas: pd.DataFrame, selected_subgroups: list[dict[str, Any]], supported: bool, runtime: float,
) -> str:
    aggregate_rows = [
        f"| {r.prediction_period} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} | {r.pr_auc:.4f} |"
        for r in aggregate.itertuples()
    ]
    architecture_rows = [
        f"| {r.architecture_id} | {r.macro_mean_precision:.4f} | {r.macro_mean_recall:.4f} | "
        f"{r.macro_mean_f1:.4f} | {r.macro_std_across_seeds_f1:.4f} | {r.eligible} |"
        for r in architectures.itertuples()
    ]
    period_rows = []
    selected_periods = periods[periods["architecture_id"].eq(selected["architecture_id"])]
    for r in selected_periods.itertuples():
        control = mulcas[mulcas["prediction_period"].eq(r.prediction_period)].iloc[0]
        period_rows.append(
            f"| {r.prediction_period} | {r.mean_precision:.4f} | {r.mean_recall:.4f} | {r.mean_f1:.4f} | "
            f"{control.precision:.4f} | {control.recall:.4f} | {control.f1:.4f} |"
        )
    subgroup_rows = []
    for row in selected_subgroups:
        mean_recall = "N/A" if row["mean_recall"] is None else f"{row['mean_recall']:.4f}"
        std_recall = "N/A" if row["std_recall"] is None else f"{row['std_recall']:.4f}"
        subgroup_rows.append(
            f"| {row['period']} | {row['subgroup']} | {row['positive_support']} | "
            f"{mean_recall} | {std_recall} |"
        )
    return "\n".join([
        "# Phase 8R2B Temporal Graph-Model Backtesting", "",
        "Phase 8R2B used only P2-P4 for forward evaluation. P5 was not loaded or evaluated.", "",
        "## Environment", "", f"- PyTorch: `{torch.__version__}`", f"- PyTorch Geometric: `{torch_geometric.__version__}`",
        f"- Device: `{'cuda' if torch.cuda.is_available() else 'cpu'}`", f"- Total GIN runtime: {runtime:.1f} seconds", "",
        "## Aggregate Baseline", "", "| Period | Precision | Recall | F1 | PR-AUC |", "| --- | ---: | ---: | ---: | ---: |",
        *aggregate_rows, "", "## GIN Architecture Summary", "",
        "| Architecture | Macro precision | Macro recall | Macro F1 | Seed F1 SD | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | --- |", *architecture_rows, "",
        f"Selected status: **{selected['selection_status']}** (`{selected['architecture_id']}`).", "",
        "## Selected GIN vs Frozen MulCas", "",
        "| Period | GIN P | GIN R | GIN F1 | MulCas P | MulCas R | MulCas F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |", *period_rows, "",
        "## Diagnostic Positive Subgroups", "",
        "These diagnostics did not participate in architecture selection.", "",
        "| Period | Subgroup | Positive support | Mean recall | Seed SD |",
        "| --- | --- | ---: | ---: | ---: |", *subgroup_rows, "",
        "## H8R2", "", f"H8R2 is **{'supported' if supported else 'not supported'}** under the predefined rule.",
        "No architecture, threshold, loss, or subgroup policy was changed after observing results.", "",
    ])


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = verify_frozen_inputs(config)
    store = FrozenGraphStore(config["graph_manifest"], config["graph_schema"], verify_chunks=True)
    frame = load_temporal_frame(config)
    labels = dict(zip(frame["row_index"].astype(int), frame["label"].astype(int), strict=True))
    graphs = store.load_rows(frame["row_index"], labels)
    statistics = pd.read_csv(config["graph_statistics"], nrows=len(frame))
    schema = json.loads(config["graph_schema"].read_text(encoding="utf-8"))
    aggregate, aggregate_names = run_aggregate_baseline(
        config, frame, graphs, statistics, schema["ordered_node_feature_names"]
    )
    aggregate.to_csv(output_dir / "aggregate_baseline_metrics.csv", index=False)
    runs, predictions, diagnostics, _ = run_gin_experiments(config, frame, graphs, output_dir)
    periods, architectures, selected = summarize_gin(runs, config)
    periods.to_csv(output_dir / "gin_period_metrics.csv", index=False)
    architectures.to_csv(output_dir / "gin_architecture_summary.csv", index=False)
    subgroups = subgroup_metrics(predictions, config["phase7e_membership"])
    subgroups.to_csv(output_dir / "gin_subgroup_metrics.csv", index=False)
    mulcas = frozen_mulcas_metrics(config["phase7a_results"])
    mulcas_macro_f1 = float(mulcas["f1"].mean())
    supported = bool(selected["selection_status"] == "eligible" and selected["macro_mean_f1"] > mulcas_macro_f1)
    runtime = float(runs["runtime_seconds"].sum())
    selected_subgroups: list[dict[str, Any]] = []
    selected_values = subgroups[subgroups["architecture_id"].eq(selected["architecture_id"])]
    for (period, subgroup), group in selected_values.groupby(["period", "subgroup"], sort=True):
        recalls = group["recall"].dropna()
        selected_subgroups.append({
            "period": period, "subgroup": subgroup, "positive_support": int(group["positive_support"].iloc[0]),
            "mean_recall": float(recalls.mean()) if len(recalls) else None,
            "std_recall": float(recalls.std(ddof=0)) if len(recalls) else None,
        })
    results = {
        "phase": "8R2B", "hypothesis": config["hypothesis"], "p5_loaded_or_evaluated": False,
        "graph_corpus_sha256": EXPECTED_CORPUS_SHA256, "graph_schema_sha256": EXPECTED_SCHEMA_SHA256,
        "frozen_artifact_hashes": frozen, "config_sha256": config["config_sha256"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_geometric": torch_geometric.__version__, "sklearn": sklearn.__version__,
                        "device": "cuda" if torch.cuda.is_available() else "cpu"},
        "aggregate_feature_dimension": len(aggregate_names), "aggregate_feature_names": aggregate_names,
        "architecture_count": len(architectures), "seeds": config["gin"]["seeds"], "run_count": len(runs),
        "selected_gin": selected, "frozen_mulcas_macro_f1": mulcas_macro_f1,
        "selected_gin_subgroup_diagnostics": selected_subgroups,
        "hypothesis_support_rule": config["selection"]["hypothesis_support"], "h8r2_supported": supported,
        "total_gin_runtime_seconds": runtime,
    }
    (output_dir / "phase8r2b_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "phase8r2b_report.md").write_text(
        build_report(aggregate, architectures, periods, selected, mulcas, selected_subgroups, supported, runtime),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    results = run(load_config(args.config))
    print(json.dumps({key: results[key] for key in ("run_count", "selected_gin", "h8r2_supported")}, indent=2))


if __name__ == "__main__":
    main()
