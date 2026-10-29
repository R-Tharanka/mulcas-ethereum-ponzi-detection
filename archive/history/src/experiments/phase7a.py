"""Phase 7A temporal robustness analysis for the frozen MulCas reconstruction."""

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
from scipy.sparse import csr_matrix, hstack

from src.evaluation.metrics import binary_classification_metrics
from src.evaluation.temporal import (
    TemporalScenario,
    build_ponzi_occurrence_periods,
    incremental_scenarios,
    validate_scenario_chronology,
)
from src.experiments.phase6a import _git_commit, sha256_file, verify_frozen_manifest
from src.features.developer import ChronologicalCreatorTransformer
from src.features.opcode import (
    make_opcode_term_count_vectorizer,
    make_opcode_tfidf_vectorizer,
    make_paper_aligned_ngram_count_vectorizer,
    make_paper_aligned_ngram_tfidf_vectorizer,
)
from src.models.baselines import build_baseline_model
from src.models.mulcas import MulCasCascade, ViewerPrediction, ViewerSpec
from src.models.run_mulcas import positive_probability
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "phase7" / "temporal_robustness.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load Phase 7A and reject settings that could alter frozen behavior."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset",
        "paper_feature_config",
        "paper_model_config",
        "paper_mulcas_config",
        "frozen_manifest",
        "output_dir",
    ):
        config[key] = _resolve(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    protocol = config["protocol"]
    if protocol["optimization"] != "prohibited":
        raise ValueError("Phase 7A optimization must remain prohibited.")
    if protocol["phase6_interventions_enabled"]:
        raise ValueError("Phase 6 interventions must not be enabled in Phase 7A.")
    return config


def load_full_dataset(path: Path) -> pd.DataFrame:
    """Load only fields needed by the frozen static temporal analysis."""
    frame = pd.read_csv(
        path,
        usecols=["address", "opcode", "creator", "createdBlockNumber", "label"],
        low_memory=False,
    )
    frame["row_index"] = frame.index.astype(int)
    if frame[["address", "creator", "createdBlockNumber", "label"]].isna().any().any():
        raise ValueError("Required Phase 7A metadata contains null values.")
    frame["opcode"] = frame["opcode"].fillna("").astype(str)
    return frame


def _load_frozen_configs(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    feature = yaml.safe_load(config["paper_feature_config"].read_text(encoding="utf-8"))
    model = yaml.safe_load(config["paper_model_config"].read_text(encoding="utf-8"))
    mulcas = yaml.safe_load(config["paper_mulcas_config"].read_text(encoding="utf-8"))
    if not mulcas.get("frozen_baseline"):
        raise ValueError("The configured MulCas reconstruction is not frozen.")
    if model.get("append_creator_to_static_views") is not True:
        raise ValueError("Frozen views must include the creator feature.")
    if any(values.get("class_weight") is not None for name, values in model["models"].items() if name in {"ridge", "svm"}):
        raise ValueError("Class weighting is not allowed in Phase 7A.")
    if float(model["models"]["xgboost"].get("scale_pos_weight", 1.0)) != 1.0:
        raise ValueError("XGBoost scale_pos_weight must remain neutral.")
    return feature, model, mulcas


def build_historical_feature_matrices(
    train: pd.DataFrame,
    target: pd.DataFrame,
    feature_views: list[str],
    *,
    ngram_min_df: int,
) -> tuple[dict[str, tuple[csr_matrix, csr_matrix]], dict[str, Any], np.ndarray]:
    """Fit every configured view on historical rows and transform the target."""
    factories = {
        "opcode_term_count": make_opcode_term_count_vectorizer,
        "opcode_tfidf": make_opcode_tfidf_vectorizer,
        "opcode_ngram_count_1_3_df_filtered": lambda: make_paper_aligned_ngram_count_vectorizer(ngram_min_df),
        "opcode_ngram_tfidf_1_3_df_filtered": lambda: make_paper_aligned_ngram_tfidf_vectorizer(ngram_min_df),
    }
    unknown = set(feature_views).difference(factories)
    if unknown:
        raise ValueError(f"Non-frozen MulCas feature views requested: {sorted(unknown)}")

    creator = ChronologicalCreatorTransformer()
    train_creator = csr_matrix(creator.fit_transform(train))
    target_creator_values = creator.transform(
        target.loc[:, ["creator", "createdBlockNumber"]]
    )
    target_creator = csr_matrix(target_creator_values)
    train_documents = train["opcode"].tolist()
    target_documents = target["opcode"].tolist()
    matrices: dict[str, tuple[csr_matrix, csr_matrix]] = {}
    dimensions: dict[str, int] = {}
    vocabularies: dict[str, int] = {}
    for feature_view in feature_views:
        transformer = factories[feature_view]()
        train_static = transformer.fit_transform(train_documents).tocsr()
        target_static = transformer.transform(target_documents).tocsr()
        train_matrix = hstack([train_static, train_creator], format="csr")
        target_matrix = hstack([target_static, target_creator], format="csr")
        matrices[feature_view] = (train_matrix, target_matrix)
        dimensions[feature_view] = int(train_matrix.shape[1])
        vocabularies[feature_view] = int(len(transformer.vocabulary_))

    audit = {
        "feature_fit_rows": int(len(train)),
        "feature_fit_block_range": [
            int(train["createdBlockNumber"].min()),
            int(train["createdBlockNumber"].max()),
        ],
        "target_transform_rows": int(len(target)),
        "target_block_range": [
            int(target["createdBlockNumber"].min()),
            int(target["createdBlockNumber"].max()),
        ],
        "historical_only": bool(
            train["createdBlockNumber"].max() < target["createdBlockNumber"].min()
        ),
        "creator_positive_state_size": int(len(creator.positive_creators_)),
        "feature_dimensions": dimensions,
        "vocabulary_sizes": vocabularies,
        "word2vec_fit": False,
        "word2vec_reason": "not present in the frozen MulCas viewer sequence",
    }
    return matrices, audit, target_creator_values.reshape(-1)


def _viewer_output(model: Any, features: csr_matrix, indices: np.ndarray) -> ViewerPrediction:
    return ViewerPrediction(
        probability=pd.Series(positive_probability(model, features), index=indices),
        prediction=pd.Series(np.asarray(model.predict(features), dtype=int), index=indices),
    )


def fit_predict_frozen_mulcas(
    train: pd.DataFrame,
    target: pd.DataFrame,
    matrices: dict[str, tuple[csr_matrix, csr_matrix]],
    model_config: dict[str, Any],
    mulcas_config: dict[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, dict[str, ViewerPrediction]]]:
    """Fit the exact frozen residual cascade and predict one later period."""
    labels = train["label"].to_numpy(dtype=int)
    target_indices = target["row_index"].to_numpy(dtype=int)
    cascade_outputs: dict[str, ViewerPrediction] = {}
    voting_outputs: dict[str, ViewerPrediction] = {}
    training_trace: list[dict[str, Any]] = []
    residual_positions = np.arange(len(train), dtype=int)
    cascade_viewers = mulcas_config["viewer_sequence"][: int(mulcas_config["complexity_control_n"])]

    for stage, viewer in enumerate(cascade_viewers, start=1):
        train_matrix, target_matrix = matrices[viewer["feature_view"]]
        model = build_baseline_model(
            viewer["model"], model_config["models"][viewer["model"]], seed
        )
        model.fit(train_matrix[residual_positions], labels[residual_positions])
        train_probabilities = positive_probability(model, train_matrix[residual_positions])
        confident = (
            train_probabilities < float(viewer["negative_threshold"])
        ) | (train_probabilities > float(viewer["positive_threshold"]))
        training_trace.append(
            {
                "stage": stage,
                "viewer": viewer["name"],
                "training_rows_in": int(len(residual_positions)),
                "positive_training_rows_in": int(labels[residual_positions].sum()),
                "confident_training_rows": int(confident.sum()),
                "training_rows_out": int((~confident).sum()),
                "positive_training_rows_out": int(labels[residual_positions][~confident].sum()),
            }
        )
        residual_positions = residual_positions[~confident]
        cascade_outputs[viewer["name"]] = _viewer_output(
            model, target_matrix, target_indices
        )

    for viewer in mulcas_config["viewer_sequence"]:
        train_matrix, target_matrix = matrices[viewer["feature_view"]]
        model = build_baseline_model(
            viewer["model"], model_config["models"][viewer["model"]], seed
        )
        model.fit(train_matrix, labels)
        voting_outputs[viewer["name"]] = _viewer_output(
            model, target_matrix, target_indices
        )

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


def _period_summary(frame: pd.DataFrame) -> dict[str, Any]:
    positives = int(frame["label"].sum())
    negatives = int(len(frame) - positives)
    return {
        "row_count": int(len(frame)),
        "positive_count": positives,
        "negative_count": negatives,
        "negative_to_positive_ratio": float(negatives / positives),
        "block_range": [
            int(frame["createdBlockNumber"].min()),
            int(frame["createdBlockNumber"].max()),
        ],
    }


def evaluate_scenario(
    assigned: pd.DataFrame,
    scenario: TemporalScenario,
    feature_config: dict[str, Any],
    model_config: dict[str, Any],
    mulcas_config: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run one predeclared forward-only temporal evaluation scenario."""
    validate_scenario_chronology(assigned, scenario)
    train = assigned.loc[assigned["period"].isin(scenario.train_periods)].copy()
    target = assigned.loc[assigned["period"] == scenario.prediction_period].copy()
    train = train.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
    target = target.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
    feature_views = list(
        dict.fromkeys(item["feature_view"] for item in mulcas_config["viewer_sequence"])
    )
    matrices, feature_audit, creator_state = build_historical_feature_matrices(
        train,
        target,
        feature_views,
        ngram_min_df=int(feature_config["ngram"]["min_df"]),
    )
    started = time.perf_counter()
    predictions, traces, training_trace, viewer_outputs = fit_predict_frozen_mulcas(
        train, target, matrices, model_config, mulcas_config, seed=seed
    )
    runtime = time.perf_counter() - started

    prediction_rows = predictions.rename(
        columns={
            "sample_index": "row_index",
            "prediction": "final_prediction",
            "probability": "final_score",
            "handled_by": "accepted_viewer",
        }
    ).merge(
        target[["row_index", "address", "createdBlockNumber", "label", "period"]],
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    prediction_rows["scenario_id"] = scenario.scenario_id
    prediction_rows["train_periods"] = "+".join(scenario.train_periods)
    prediction_rows["reached_voting_stage"] = prediction_rows["accepted_viewer"].eq("vote")
    prediction_rows["cascade_stage"] = prediction_rows["accepted_viewer"].map(
        {
            viewer["name"]: stage
            for stage, viewer in enumerate(
                mulcas_config["viewer_sequence"][: int(mulcas_config["complexity_control_n"])],
                start=1,
            )
        }
    ).fillna(int(mulcas_config["complexity_control_n"]) + 1).astype(int)
    prediction_rows["creator_history_feature"] = creator_state
    for phase_name, outputs in viewer_outputs.items():
        for viewer_name, output in outputs.items():
            lookup = prediction_rows["row_index"]
            prediction_rows[f"{phase_name}__{viewer_name}__score"] = lookup.map(output.probability)
            prediction_rows[f"{phase_name}__{viewer_name}__prediction"] = lookup.map(output.prediction)

    metrics = binary_classification_metrics(
        prediction_rows["label"].to_numpy(dtype=int),
        prediction_rows["final_prediction"].to_numpy(dtype=int),
        prediction_rows["final_score"].to_numpy(dtype=float),
    )
    routing_counts = {
        str(key): int(value)
        for key, value in prediction_rows["accepted_viewer"].value_counts().items()
    }
    scenario_result = {
        "scenario_id": scenario.scenario_id,
        "train_periods": list(scenario.train_periods),
        "prediction_period": scenario.prediction_period,
        "train": _period_summary(train),
        "prediction": _period_summary(target),
        "metrics": metrics,
        "routing_counts": routing_counts,
        "feature_fit_audit": feature_audit,
        "cascade_training_trace": training_trace,
        "runtime_seconds": float(runtime),
    }

    trace_rows = traces.rename(columns={"sample_index": "row_index"}).merge(
        prediction_rows[
            [
                "row_index",
                "address",
                "period",
                "label",
                "final_prediction",
                "final_score",
                "accepted_viewer",
                "reached_voting_stage",
                "creator_history_feature",
            ]
        ],
        on="row_index",
        how="left",
        validate="many_to_one",
    )
    trace_rows["scenario_id"] = scenario.scenario_id
    return scenario_result, prediction_rows, trace_rows


def run_phase7a(config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Execute all four fixed temporal robustness scenarios."""
    set_random_seed(int(config["seed"]))
    frozen_hashes = verify_frozen_manifest(config["frozen_manifest"])
    feature_config, model_config, mulcas_config = _load_frozen_configs(config)
    frame = load_full_dataset(config["dataset"])
    assigned, definitions = build_ponzi_occurrence_periods(
        frame,
        positive_ends=config["periods"]["positive_ends"],
        expected_positives=int(config["periods"]["expected_total_positives"]),
    )
    configured = [
        (tuple(item["train_periods"]), item["prediction_period"])
        for item in config["scenarios"]
    ]
    scenarios = incremental_scenarios()
    expected = [(item.train_periods, item.prediction_period) for item in scenarios]
    if configured != expected:
        raise ValueError("Configured scenarios differ from the predeclared Phase 7A scenarios.")

    scenario_results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    for scenario in scenarios:
        result, predictions, traces = evaluate_scenario(
            assigned,
            scenario,
            feature_config,
            model_config,
            mulcas_config,
            seed=int(config["seed"]),
        )
        scenario_results.append(result)
        prediction_frames.append(predictions)
        trace_frames.append(traces)

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
    )
    changes: list[dict[str, Any]] = []
    for previous, current in zip(scenario_results, scenario_results[1:], strict=False):
        changes.append(
            {
                "from_period": previous["prediction_period"],
                "to_period": current["prediction_period"],
                **{
                    f"delta_{metric}": float(
                        current["metrics"][metric] - previous["metrics"][metric]
                    )
                    for metric in metric_names
                },
            }
        )

    results = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "7A",
        "research_question": "RQ5: Does reconstructed MulCas retain effectiveness on later contracts?",
        "analysis_only": True,
        "optimization_performed": False,
        "git_commit": _git_commit(),
        "random_seed": int(config["seed"]),
        "dataset_sha256": sha256_file(config["dataset"]),
        "config_sha256": config["config_sha256"],
        "frozen_artifact_hashes": frozen_hashes,
        "frozen_mulcas_config_sha256": sha256_file(config["paper_mulcas_config"]),
        "period_definitions": definitions,
        "scenarios": scenario_results,
        "descriptive_period_changes": changes,
        "published_paper_reference": config["paper_reference"],
        "known_comparability_limits": [
            "The exact N-gram document-frequency cutoff is not reported by the paper.",
            "The publication does not fully specify all model/library settings.",
            "Whole-block assignment can differ from occurrence slicing when positives share a boundary block.",
            "Paper metrics absent from Table 4 were not inferred.",
        ],
    }
    return results, pd.concat(prediction_frames, ignore_index=True), pd.concat(trace_frames, ignore_index=True)


def _report(results: dict[str, Any]) -> str:
    period_rows = [
        "| Period | Blocks | Rows | Positive | Negative | Neg:Pos | Boundary delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["period_definitions"]:
        period_rows.append(
            f"| {row['period']} | {row['block_min']}-{row['block_max']} | {row['row_count']} | "
            f"{row['positive_count']} | {row['negative_count']} | {row['negative_to_positive_ratio']:.3f} | "
            f"{row['positive_count_difference']:+d} |"
        )
    metric_rows = [
        "| Predict | Train periods | Accuracy | Balanced acc. | Precision | Recall | F1 | PR-AUC | ROC-AUC | TN | FP | FN | TP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    routing_rows = [
        "| Predict | Routing distribution |",
        "| --- | --- |",
    ]
    for scenario in results["scenarios"]:
        metrics = scenario["metrics"]
        metric_rows.append(
            f"| {scenario['prediction_period']} | {' + '.join(scenario['train_periods'])} | "
            f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['pr_auc']:.4f} | "
            f"{metrics['roc_auc']:.4f} | {metrics['tn']} | {metrics['fp']} | {metrics['fn']} | {metrics['tp']} |"
        )
        routing = ", ".join(f"{key}: {value}" for key, value in scenario["routing_counts"].items())
        routing_rows.append(f"| {scenario['prediction_period']} | {routing} |")
    comparison_rows = [
        "| Period | Metric | Published paper | Current reconstruction | Difference |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    published = results["published_paper_reference"]["mulcas_metrics"]
    for scenario in results["scenarios"]:
        period = scenario["prediction_period"]
        for metric in ("precision", "recall", "f1"):
            paper_value = float(published[period][metric])
            current_value = float(scenario["metrics"][metric])
            comparison_rows.append(
                f"| {period} | {metric} | {paper_value:.4f} | {current_value:.4f} | "
                f"{current_value - paper_value:+.4f} |"
            )
    last = results["scenarios"][-1]["metrics"]
    first = results["scenarios"][0]["metrics"]
    trend = (
        f"From P2 to P5, precision changed by {last['precision'] - first['precision']:+.4f}, "
        f"recall by {last['recall'] - first['recall']:+.4f}, and F1 by {last['f1'] - first['f1']:+.4f}. "
        "These are descriptive temporal variations and do not by themselves prove concept drift."
    )
    return (
        "# Phase 7A: Temporal Robustness and Model Aging\n\n"
        "Phase 7A evaluates the unchanged frozen Phase 5.5 MulCas reconstruction. No model, feature, threshold, routing, or ensemble setting was optimized from P2-P5 results.\n\n"
        "## Period Definitions\n\n" + "\n".join(period_rows) + "\n\n"
        "Whole creation blocks remain intact. `Boundary delta` records any positive-count difference caused by contracts sharing a boundary block. The paper reports 5,990 P0-P4 rows and 508 P5 rows; this reconstruction measures 5,989 and 509 under the explicit whole-block policy. The publication does not expose row membership, so the exact source of this one-row difference cannot be resolved.\n\n"
        "## Current Reconstruction\n\n" + "\n".join(metric_rows) + "\n\n"
        "## Routing\n\n" + "\n".join(routing_rows) + "\n\n"
        "## Temporal Interpretation\n\n" + trend + "\n\n"
        "Performance is not monotonic: recall is 0.2600, 0.1400, 0.9200, and 0.4219 across P2-P5. P2 is also unusually imbalanced at 101.82 negatives per positive, making its high accuracy misleading. P5 retains high precision but misses 37 of 64 positives.\n\n"
        "The expanding training history changes together with the prediction period, so these scenarios describe operational forward performance; they do not isolate time from training-set-size effects.\n\n"
        "## Published Paper Comparison\n\n"
        f"**PUBLISHED PAPER:** {results['published_paper_reference']['citation']}. "
        f"{results['published_paper_reference']['note']}\n\n"
        + "\n".join(comparison_rows) + "\n\n"
        "**CURRENT RECONSTRUCTION:** the exact measured values are reported above. Direct exact replication is not claimed because the N-gram cutoff, full model settings, and boundary handling are under-specified.\n"
    )


def save_results(
    results: dict[str, Any],
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write all Phase 7A machine-readable and narrative artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "phase7a_results.json",
        "periods": output_dir / "phase7a_period_definitions.json",
        "metrics": output_dir / "phase7a_period_metrics.csv",
        "predictions": output_dir / "phase7a_predictions.csv",
        "routing": output_dir / "phase7a_routing_diagnostics.csv",
        "report": output_dir / "phase7a_report.md",
    }
    paths["results"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    paths["periods"].write_text(
        json.dumps(results["period_definitions"], indent=2), encoding="utf-8"
    )
    metric_rows = []
    for scenario in results["scenarios"]:
        metric_rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "train_periods": "+".join(scenario["train_periods"]),
                "prediction_period": scenario["prediction_period"],
                **{f"train_{key}": value for key, value in scenario["train"].items()},
                **{f"prediction_{key}": value for key, value in scenario["prediction"].items()},
                **scenario["metrics"],
                "runtime_seconds": scenario["runtime_seconds"],
            }
        )
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    diagnostics.to_csv(paths["routing"], index=False)
    paths["report"].write_text(_report(results), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    results, predictions, diagnostics = run_phase7a(config)
    for name, path in save_results(results, predictions, diagnostics, config["output_dir"]).items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
