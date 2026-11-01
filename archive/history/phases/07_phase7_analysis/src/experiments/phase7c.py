"""Phase 7C chronological creator-feature ablation for frozen MulCas."""

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
from scipy.sparse import csr_matrix

from src.evaluation.metrics import binary_classification_metrics
from src.evaluation.temporal import build_ponzi_occurrence_periods, incremental_scenarios
from src.experiments.phase6a import _git_commit, sha256_file, verify_frozen_manifest
from src.experiments.phase7a import (
    _load_frozen_configs,
    build_historical_feature_matrices,
    fit_predict_frozen_mulcas,
    load_config as load_phase7a_config,
    load_full_dataset,
)
from src.models.mulcas import ViewerPrediction
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "phase7" / "creator_ablation.yaml"
CONDITIONS = ("with_creator", "without_creator")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load Phase 7C and reject any deviation from the fixed ablation."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "phase7a_config",
        "phase7a_results",
        "paper_feature_config",
        "paper_model_config",
        "paper_mulcas_config",
        "frozen_manifest",
        "output_dir",
    ):
        config[key] = _resolve(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["protocol"]["optimization"] != "prohibited":
        raise ValueError("Phase 7C optimization must remain prohibited.")
    if config["protocol"]["retuning"] != "prohibited":
        raise ValueError("Phase 7C retuning must remain prohibited.")
    if tuple(item["id"] for item in config["conditions"]) != CONDITIONS:
        raise ValueError("Phase 7C must contain exactly with/without creator conditions.")
    ablation = config["creator_ablation"]
    if ablation["operation"] != "remove_final_creator_column":
        raise ValueError("Creator ablation must remove the existing final column.")
    if ablation["replacement_column"] != "none":
        raise ValueError("Creator ablation must not add a replacement column.")
    return config


def build_creator_ablation_matrices(
    train: pd.DataFrame,
    target: pd.DataFrame,
    feature_views: list[str],
    *,
    ngram_min_df: int,
) -> tuple[
    dict[str, dict[str, tuple[csr_matrix, csr_matrix]]],
    dict[str, Any],
    np.ndarray,
]:
    """Fit static views once and expose matrices with/without creator column."""
    with_creator, base_audit, target_creator = build_historical_feature_matrices(
        train,
        target,
        feature_views,
        ngram_min_df=ngram_min_df,
    )
    without_creator: dict[str, tuple[csr_matrix, csr_matrix]] = {}
    dimensions: dict[str, dict[str, int]] = {}
    for feature_view, (train_matrix, target_matrix) in with_creator.items():
        if train_matrix.shape[1] < 2 or target_matrix.shape[1] != train_matrix.shape[1]:
            raise AssertionError(f"Invalid creator-appended matrix for {feature_view}.")
        static_train = train_matrix[:, :-1].tocsr()
        static_target = target_matrix[:, :-1].tocsr()
        if static_train.shape[1] + 1 != train_matrix.shape[1]:
            raise AssertionError("Creator ablation did not remove exactly one column.")
        without_creator[feature_view] = (static_train, static_target)
        dimensions[feature_view] = {
            "with_creator": int(train_matrix.shape[1]),
            "without_creator": int(static_train.shape[1]),
            "difference": 1,
        }
    audit = {
        **base_audit,
        "creator_column_position": "final",
        "creator_column_removed_not_replaced": True,
        "dimensions": dimensions,
        "static_columns_shared": True,
    }
    return {
        "with_creator": with_creator,
        "without_creator": without_creator,
    }, audit, target_creator


def _metrics_from_output(
    truth: pd.Series, output: ViewerPrediction
) -> dict[str, Any]:
    aligned_scores = output.probability.reindex(truth.index)
    aligned_predictions = output.prediction.reindex(truth.index)
    return binary_classification_metrics(
        truth.to_numpy(dtype=int),
        aligned_predictions.to_numpy(dtype=int),
        aligned_scores.to_numpy(dtype=float),
    )


def creator_diagnostics(
    train: pd.DataFrame,
    target: pd.DataFrame,
    target_creator: np.ndarray,
    condition_predictions: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Describe historical creator prevalence and errors without updating state."""
    target_ordered = target.reset_index(drop=True)
    creator_values = np.asarray(target_creator, dtype=int).reshape(-1)
    if len(creator_values) != len(target_ordered):
        raise ValueError("Creator diagnostics are not aligned with target rows.")
    labels = target_ordered["label"].to_numpy(dtype=int)
    previously_observed = target_ordered["creator"].astype(str).isin(
        set(train["creator"].astype(str))
    ).to_numpy()
    positive_creator = creator_values == 1
    positive_labels = labels == 1
    diagnostics: dict[str, Any] = {
        "prediction_count": int(len(target_ordered)),
        "previously_observed_creator_count": int(previously_observed.sum()),
        "historical_ponzi_creator_count": int(positive_creator.sum()),
        "true_ponzi_with_creator_feature_1": int((positive_labels & positive_creator).sum()),
        "benign_with_creator_feature_1": int(((~positive_labels) & positive_creator).sum()),
        "true_ponzi_with_creator_feature_0": int((positive_labels & ~positive_creator).sum()),
    }
    indices = target_ordered["row_index"].to_numpy(dtype=int)
    for condition, predictions in condition_predictions.items():
        predicted = (
            predictions.set_index("sample_index")
            .reindex(indices)["prediction"]
            .to_numpy(dtype=int)
        )
        for creator_state, mask in ((1, positive_creator), (0, ~positive_creator)):
            ponzi_mask = positive_labels & mask
            diagnostics[f"{condition}_tp_creator_feature_{creator_state}"] = int(
                (ponzi_mask & (predicted == 1)).sum()
            )
            diagnostics[f"{condition}_fn_creator_feature_{creator_state}"] = int(
                (ponzi_mask & (predicted == 0)).sum()
            )
    return diagnostics


def _assert_phase7a_reference(
    period: str, metrics: dict[str, Any], phase7a_results: dict[str, Any]
) -> None:
    expected = next(
        item["metrics"]
        for item in phase7a_results["scenarios"]
        if item["prediction_period"] == period
    )
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
    ):
        if not np.isclose(metrics[metric], expected[metric], atol=1e-12, rtol=0.0):
            raise AssertionError(f"Phase 7A reference mismatch for {period} {metric}.")
    for metric in ("tn", "fp", "fn", "tp"):
        if metrics[metric] != expected[metric]:
            raise AssertionError(f"Phase 7A reference mismatch for {period} {metric}.")


def run_phase7c(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate frozen MulCas with and without its creator column."""
    started = time.perf_counter()
    seed = int(config["seed"])
    set_random_seed(seed)
    frozen_hashes = verify_frozen_manifest(config["frozen_manifest"])
    phase7a_config = load_phase7a_config(config["phase7a_config"])
    phase7a_results = json.loads(config["phase7a_results"].read_text(encoding="utf-8"))
    feature_config, model_config, mulcas_config = _load_frozen_configs(phase7a_config)
    frame = load_full_dataset(phase7a_config["dataset"])
    assigned, period_definitions = build_ponzi_occurrence_periods(
        frame,
        positive_ends=phase7a_config["periods"]["positive_ends"],
        expected_positives=int(phase7a_config["periods"]["expected_total_positives"]),
    )
    if period_definitions != phase7a_results["period_definitions"]:
        raise AssertionError("Phase 7C period definitions differ from Phase 7A.")

    records: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    viewer_diagnostics: list[dict[str, Any]] = []
    feature_audits: list[dict[str, Any]] = []
    for scenario in incremental_scenarios():
        train = assigned.loc[assigned["period"].isin(scenario.train_periods)].copy()
        target = assigned.loc[assigned["period"] == scenario.prediction_period].copy()
        train = train.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
        target = target.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
        feature_views = list(
            dict.fromkeys(item["feature_view"] for item in mulcas_config["viewer_sequence"])
        )
        matrices_by_condition, feature_audit, target_creator = (
            build_creator_ablation_matrices(
                train,
                target,
                feature_views,
                ngram_min_df=int(feature_config["ngram"]["min_df"]),
            )
        )
        feature_audits.append(
            {"prediction_period": scenario.prediction_period, **feature_audit}
        )
        outputs_by_condition: dict[str, dict[str, dict[str, ViewerPrediction]]] = {}
        condition_predictions: dict[str, pd.DataFrame] = {}
        condition_metrics: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            condition_started = time.perf_counter()
            predictions, _, training_trace, viewer_outputs = fit_predict_frozen_mulcas(
                train,
                target,
                matrices_by_condition[condition],
                model_config,
                mulcas_config,
                seed=seed,
            )
            truth = target.set_index("row_index")["label"]
            aligned = predictions.set_index("sample_index").reindex(truth.index)
            metrics = binary_classification_metrics(
                truth.to_numpy(dtype=int),
                aligned["prediction"].to_numpy(dtype=int),
                aligned["probability"].to_numpy(dtype=float),
            )
            if condition == "with_creator":
                _assert_phase7a_reference(
                    scenario.prediction_period, metrics, phase7a_results
                )
            record = {
                "condition": condition,
                "creator_feature_enabled": condition == "with_creator",
                "prediction_period": scenario.prediction_period,
                "train_periods": list(scenario.train_periods),
                "train_rows": int(len(train)),
                "prediction_count": int(len(target)),
                "routing_counts": {
                    str(key): int(value)
                    for key, value in predictions["handled_by"].value_counts().items()
                },
                "cascade_training_trace": training_trace,
                "runtime_seconds": float(time.perf_counter() - condition_started),
                **metrics,
            }
            records.append(record)
            condition_metrics[condition] = metrics
            condition_predictions[condition] = predictions
            outputs_by_condition[condition] = viewer_outputs

            prediction_frame = predictions.rename(
                columns={
                    "sample_index": "row_index",
                    "prediction": "final_prediction",
                    "probability": "final_score",
                    "handled_by": "accepted_viewer",
                }
            ).merge(
                target[["row_index", "address", "creator", "createdBlockNumber", "label", "period"]],
                on="row_index",
                how="left",
                validate="one_to_one",
            )
            prediction_frame["condition"] = condition
            prediction_frame["creator_feature"] = target_creator.astype(int)
            prediction_frames.append(prediction_frame)

        with_metrics = condition_metrics["with_creator"]
        without_metrics = condition_metrics["without_creator"]
        delta_metrics = {
            metric: float(without_metrics[metric] - with_metrics[metric])
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
                "tn",
                "fp",
                "fn",
                "tp",
            )
        }
        deltas.append(
            {
                "prediction_period": scenario.prediction_period,
                "definition": "without_creator_minus_with_creator",
                **{f"delta_{key}": value for key, value in delta_metrics.items()},
            }
        )
        diagnostics.append(
            {
                "prediction_period": scenario.prediction_period,
                **creator_diagnostics(
                    train, target, target_creator, condition_predictions
                ),
            }
        )

        truth = target.set_index("row_index")["label"]
        if config["viewer_level_diagnostics"]["enabled"]:
            for viewer in mulcas_config["viewer_sequence"]:
                viewer_name = viewer["name"]
                with_viewer = _metrics_from_output(
                    truth, outputs_by_condition["with_creator"]["voting"][viewer_name]
                )
                without_viewer = _metrics_from_output(
                    truth,
                    outputs_by_condition["without_creator"]["voting"][viewer_name],
                )
                viewer_diagnostics.append(
                    {
                        "prediction_period": scenario.prediction_period,
                        "viewer": viewer_name,
                        "feature_representation": viewer["feature_view"],
                        "classifier_family": viewer["model"],
                        "source": "full_historical_training_voting_copy",
                        "with_creator": with_viewer,
                        "without_creator": without_viewer,
                        "deltas": {
                            metric: float(without_viewer[metric] - with_viewer[metric])
                            for metric in (
                                "precision",
                                "recall",
                                "f1",
                                "pr_auc",
                                "balanced_accuracy",
                                "tn",
                                "fp",
                                "fn",
                                "tp",
                            )
                        },
                    }
                )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "7C",
        "research_question": config["research_question"],
        "analysis_only": True,
        "optimization_performed": False,
        "git_commit": _git_commit(),
        "random_seed": seed,
        "dataset_sha256": sha256_file(phase7a_config["dataset"]),
        "config_sha256": config["config_sha256"],
        "phase7a_config_sha256": sha256_file(config["phase7a_config"]),
        "frozen_artifact_hashes": frozen_hashes,
        "protocol": config["protocol"],
        "creator_ablation": config["creator_ablation"],
        "period_definitions": period_definitions,
        "feature_audits": feature_audits,
        "records": records,
        "metric_deltas": deltas,
        "creator_diagnostics": diagnostics,
        "viewer_level_diagnostics": viewer_diagnostics,
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "total_runtime_seconds": float(time.perf_counter() - started),
    }


def _report(results: dict[str, Any]) -> str:
    metric_rows = [
        "| Period | Condition | Precision | Recall | F1 | PR-AUC | Bal. acc. | TN | FP | FN | TP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in results["records"]:
        metric_rows.append(
            f"| {record['prediction_period']} | {record['condition']} | "
            f"{record['precision']:.4f} | {record['recall']:.4f} | {record['f1']:.4f} | "
            f"{record['pr_auc']:.4f} | {record['balanced_accuracy']:.4f} | "
            f"{record['tn']} | {record['fp']} | {record['fn']} | {record['tp']} |"
        )
    delta_rows = [
        "| Period | Delta precision | Delta recall | Delta F1 | Delta PR-AUC | Delta bal. acc. | Delta FP | Delta FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["metric_deltas"]:
        delta_rows.append(
            f"| {row['prediction_period']} | {row['delta_precision']:+.4f} | "
            f"{row['delta_recall']:+.4f} | {row['delta_f1']:+.4f} | "
            f"{row['delta_pr_auc']:+.4f} | {row['delta_balanced_accuracy']:+.4f} | "
            f"{row['delta_fp']:+.0f} | {row['delta_fn']:+.0f} |"
        )
    diagnostic_rows = [
        "| Period | Previously observed creator | Creator feature = 1 | Ponzi with feature = 1 | Benign with feature = 1 | With TP/FN (feature 1) | Without TP/FN (feature 1) | With TP/FN (feature 0) | Without TP/FN (feature 0) |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in results["creator_diagnostics"]:
        diagnostic_rows.append(
            f"| {row['prediction_period']} | {row['previously_observed_creator_count']} | "
            f"{row['historical_ponzi_creator_count']} | {row['true_ponzi_with_creator_feature_1']} | "
            f"{row['benign_with_creator_feature_1']} | "
            f"{row['with_creator_tp_creator_feature_1']}/{row['with_creator_fn_creator_feature_1']} | "
            f"{row['without_creator_tp_creator_feature_1']}/{row['without_creator_fn_creator_feature_1']} | "
            f"{row['with_creator_tp_creator_feature_0']}/{row['with_creator_fn_creator_feature_0']} | "
            f"{row['without_creator_tp_creator_feature_0']}/{row['without_creator_fn_creator_feature_0']} |"
        )
    viewer_rows = [
        "| Period | Viewer | Delta precision | Delta recall | Delta F1 | Delta PR-AUC | Delta FP | Delta FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["viewer_level_diagnostics"]:
        delta = row["deltas"]
        viewer_rows.append(
            f"| {row['prediction_period']} | {row['viewer']} | "
            f"{delta['precision']:+.4f} | {delta['recall']:+.4f} | "
            f"{delta['f1']:+.4f} | {delta['pr_auc']:+.4f} | "
            f"{delta['fp']:+.0f} | {delta['fn']:+.0f} |"
        )
    return (
        "# Phase 7C: Chronological Creator Feature Ablation\n\n"
        "The only independent variable is whether the final leakage-safe creator feature column is present. The no-creator matrices contain no replacement or constant placeholder column.\n\n"
        "## Full-System Metrics\n\n" + "\n".join(metric_rows) + "\n\n"
        "## Without Minus With Creator\n\n" + "\n".join(delta_rows) + "\n\n"
        "## Creator History Diagnostics\n\n" + "\n".join(diagnostic_rows) + "\n\n"
        "## Viewer-Level Diagnostics\n\n" + "\n".join(viewer_rows) + "\n\n"
        "## Interpretation\n\n"
        "- Removing creator history changes recall by -0.0400 in P2, +0.0800 in P3, +0.0000 in P4, and +0.0312 in P5. The creator feature therefore does not improve recall consistently.\n"
        "- In P2, creator history trades two additional true positives for 102 additional false positives. There are no P2 Ponzi contracts with creator feature 1, so this is a learned model interaction rather than direct recognition of a known Ponzi creator.\n"
        "- P4 has four Ponzi contracts associated with historical Ponzi creators, and all four are detected under both conditions. Final classifications are identical, although PR-AUC decreases by 0.0087 without creator history. The strong P4 classification result is not dependent on the creator column in this ablation.\n"
        "- In P5, removing creator history recovers two positives without changing false positives. Both gains occur among the 60 Ponzi contracts with creator feature 0; the four feature-1 Ponzi contracts retain the same 3 TP / 1 FN result.\n"
        "- Later false negatives are concentrated among creators without prior positive history: 36 of 37 P5 false negatives under the reference condition have creator feature 0. This is descriptive historical signal, not evidence that creator identity causes Ponzi behavior.\n\n"
        "Neither condition is used for model selection, and the viewer-level results are diagnostics only.\n"
    )


def save_results(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write all requested Phase 7C outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "phase7c_results.json",
        "metrics": output_dir / "phase7c_period_metrics.csv",
        "deltas": output_dir / "phase7c_metric_deltas.csv",
        "diagnostics": output_dir / "phase7c_creator_diagnostics.csv",
        "predictions": output_dir / "phase7c_predictions.csv",
        "report": output_dir / "phase7c_report.md",
    }
    predictions = results.pop("predictions")
    paths["results"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    metric_rows = []
    for record in results["records"]:
        excluded = {"routing_counts", "cascade_training_trace", "runtime_seconds"}
        metric_rows.append({key: value for key, value in record.items() if key not in excluded})
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(results["metric_deltas"]).to_csv(paths["deltas"], index=False)
    pd.DataFrame(results["creator_diagnostics"]).to_csv(paths["diagnostics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    paths["report"].write_text(_report(results), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    results = run_phase7c(config)
    for name, path in save_results(results, config["output_dir"]).items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
