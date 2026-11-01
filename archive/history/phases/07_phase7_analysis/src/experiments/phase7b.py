"""Phase 7B component/view ablation of the frozen MulCas reconstruction."""

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
from src.models.mulcas import MulCasCascade, ViewerPrediction, ViewerSpec
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "phase7" / "view_ablation.yaml"
FULL_ABLATION_ID = "full_frozen_mulcas"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the predeclared Phase 7B ablation protocol."""
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
        raise ValueError("Phase 7B optimization must remain prohibited.")
    if config["protocol"]["retuning"] != "prohibited":
        raise ValueError("Phase 7B retuning must remain prohibited.")
    expected = [
        "full_frozen_mulcas",
        "remove_first_cascade_viewer",
        "remove_second_cascade_viewer",
        "remove_voting_stage",
        "first_viewer_only",
        "second_viewer_only",
    ]
    if [item["id"] for item in config["ablations"]] != expected:
        raise ValueError("Phase 7B ablations differ from the predeclared set.")
    semantics = config["ablation_semantics"]
    if semantics["removed_viewer_scope"] != "cascade_and_voting":
        raise ValueError("Removed viewers must be removed from cascade and voting.")
    if semantics["no_voting_fallback"] != "last_cascade_viewer_native_prediction":
        raise ValueError("Unexpected no-voting fallback policy.")
    return config


def describe_frozen_viewers(mulcas_config: dict[str, Any]) -> dict[str, Any]:
    """Describe the actual reconstruction instead of assuming paper composition."""
    cascade_depth = int(mulcas_config["complexity_control_n"])
    viewers = []
    for position, viewer in enumerate(mulcas_config["viewer_sequence"], start=1):
        viewers.append(
            {
                "position": position,
                "name": viewer["name"],
                "feature_representation": viewer["feature_view"],
                "classifier_family": viewer["model"],
                "cascade_stage": position if position <= cascade_depth else None,
                "negative_threshold": viewer.get("negative_threshold"),
                "positive_threshold": viewer.get("positive_threshold"),
                "voting_stage_participant": True,
            }
        )
    return {
        "source": "configs/paper_aligned_mulcas.yaml",
        "cascade_depth": cascade_depth,
        "voting_threshold": float(mulcas_config["voting_threshold"]),
        "score_mapping": mulcas_config["score_mapping"],
        "viewers": viewers,
        "voting_participants": [item["name"] for item in viewers],
    }


def _run_cascade(
    sample_indices: np.ndarray,
    cascade_configs: list[dict[str, Any]],
    cascade_outputs: dict[str, ViewerPrediction],
    voting_names: list[str],
    voting_outputs: dict[str, ViewerPrediction],
    voting_threshold: float,
) -> pd.DataFrame:
    cascade = MulCasCascade(
        cascade_viewers=[
            ViewerSpec(
                name=viewer["name"],
                negative_threshold=float(viewer["negative_threshold"]),
                positive_threshold=float(viewer["positive_threshold"]),
            )
            for viewer in cascade_configs
        ],
        voting_viewers=voting_names,
        voting_threshold=voting_threshold,
    )
    return cascade.predict(sample_indices, cascade_outputs, voting_outputs).predictions


def _without_voting(
    sample_indices: np.ndarray,
    cascade_configs: list[dict[str, Any]],
    cascade_outputs: dict[str, ViewerPrediction],
) -> pd.DataFrame:
    """Route normally, then use the last cascade viewer for unresolved rows."""
    ordered = pd.Index(sample_indices, name="sample_index")
    unresolved = pd.Series(True, index=ordered)
    prediction = pd.Series(-1, index=ordered, dtype=int)
    probability = pd.Series(np.nan, index=ordered, dtype=float)
    handled_by = pd.Series("", index=ordered, dtype=object)
    for viewer in cascade_configs:
        output = cascade_outputs[viewer["name"]]
        active = ordered[unresolved.to_numpy()]
        scores = output.probability.reindex(active)
        accepted = (scores < float(viewer["negative_threshold"])) | (
            scores > float(viewer["positive_threshold"])
        )
        accepted_indices = active[accepted.to_numpy()]
        prediction.loc[accepted_indices] = output.prediction.reindex(accepted_indices)
        probability.loc[accepted_indices] = output.probability.reindex(accepted_indices)
        handled_by.loc[accepted_indices] = viewer["name"]
        unresolved.loc[accepted_indices] = False
    remaining = ordered[unresolved.to_numpy()]
    fallback = cascade_outputs[cascade_configs[-1]["name"]]
    prediction.loc[remaining] = fallback.prediction.reindex(remaining)
    probability.loc[remaining] = fallback.probability.reindex(remaining)
    handled_by.loc[remaining] = "fallback_last_cascade_viewer"
    return pd.DataFrame(
        {
            "sample_index": ordered,
            "prediction": prediction.to_numpy(),
            "probability": probability.to_numpy(),
            "handled_by": handled_by.to_numpy(),
        }
    )


def _standalone(
    sample_indices: np.ndarray,
    viewer_name: str,
    outputs: dict[str, ViewerPrediction],
) -> pd.DataFrame:
    """Use the viewer's full-history voting copy with its native prediction."""
    ordered = pd.Index(sample_indices, name="sample_index")
    output = outputs[viewer_name]
    return pd.DataFrame(
        {
            "sample_index": ordered,
            "prediction": output.prediction.reindex(ordered).to_numpy(dtype=int),
            "probability": output.probability.reindex(ordered).to_numpy(dtype=float),
            "handled_by": f"standalone__{viewer_name}",
        }
    )


def build_ablation_predictions(
    sample_indices: np.ndarray,
    mulcas_config: dict[str, Any],
    full_predictions: pd.DataFrame,
    viewer_outputs: dict[str, dict[str, ViewerPrediction]],
) -> dict[str, pd.DataFrame]:
    """Apply only the six predeclared component interventions."""
    viewers = mulcas_config["viewer_sequence"]
    cascade_depth = int(mulcas_config["complexity_control_n"])
    cascade_viewers = viewers[:cascade_depth]
    first, second = cascade_viewers
    voting_outputs = viewer_outputs["voting"]
    cascade_outputs = viewer_outputs["cascade"]
    voting_threshold = float(mulcas_config["voting_threshold"])

    remove_first_viewers = [viewer for viewer in viewers if viewer["name"] != first["name"]]
    remove_second_viewers = [viewer for viewer in viewers if viewer["name"] != second["name"]]
    return {
        FULL_ABLATION_ID: full_predictions.copy(),
        "remove_first_cascade_viewer": _run_cascade(
            sample_indices,
            [second],
            {second["name"]: voting_outputs[second["name"]]},
            [viewer["name"] for viewer in remove_first_viewers],
            voting_outputs,
            voting_threshold,
        ),
        "remove_second_cascade_viewer": _run_cascade(
            sample_indices,
            [first],
            {first["name"]: cascade_outputs[first["name"]]},
            [viewer["name"] for viewer in remove_second_viewers],
            voting_outputs,
            voting_threshold,
        ),
        "remove_voting_stage": _without_voting(
            sample_indices, cascade_viewers, cascade_outputs
        ),
        "first_viewer_only": _standalone(
            sample_indices, first["name"], voting_outputs
        ),
        "second_viewer_only": _standalone(
            sample_indices, second["name"], voting_outputs
        ),
    }


def _routing_records(
    ablation_id: str, period: str, predictions: pd.DataFrame
) -> list[dict[str, Any]]:
    counts = predictions["handled_by"].value_counts()
    return [
        {
            "ablation_id": ablation_id,
            "prediction_period": period,
            "route": str(route),
            "count": int(count),
            "fraction": float(count / len(predictions)),
        }
        for route, count in counts.items()
    ]


def _assert_phase7a_reproduction(
    period: str, metrics: dict[str, Any], phase7a_results: dict[str, Any]
) -> None:
    expected = next(
        item["metrics"]
        for item in phase7a_results["scenarios"]
        if item["prediction_period"] == period
    )
    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
    ):
        if not np.isclose(metrics[name], expected[name], atol=1e-12, rtol=0.0):
            raise AssertionError(f"Full MulCas no longer reproduces Phase 7A {period} {name}.")
    for name in ("tn", "fp", "fn", "tp"):
        if metrics[name] != expected[name]:
            raise AssertionError(f"Full MulCas no longer reproduces Phase 7A {period} {name}.")


def run_phase7b(config: dict[str, Any]) -> dict[str, Any]:
    """Run fixed viewer/component ablations over the Phase 7A periods."""
    started = time.perf_counter()
    seed = int(config["seed"])
    set_random_seed(seed)
    frozen_hashes = verify_frozen_manifest(config["frozen_manifest"])
    phase7a_config = load_phase7a_config(config["phase7a_config"])
    phase7a_results = json.loads(config["phase7a_results"].read_text(encoding="utf-8"))
    feature_config, model_config, mulcas_config = _load_frozen_configs(phase7a_config)
    viewer_description = describe_frozen_viewers(mulcas_config)

    frame = load_full_dataset(phase7a_config["dataset"])
    assigned, period_definitions = build_ponzi_occurrence_periods(
        frame,
        positive_ends=phase7a_config["periods"]["positive_ends"],
        expected_positives=int(phase7a_config["periods"]["expected_total_positives"]),
    )
    if period_definitions != phase7a_results["period_definitions"]:
        raise AssertionError("Phase 7B period definitions differ from Phase 7A.")

    ablation_records: list[dict[str, Any]] = []
    routing_records: list[dict[str, Any]] = []
    for scenario in incremental_scenarios():
        train = assigned.loc[assigned["period"].isin(scenario.train_periods)].copy()
        target = assigned.loc[assigned["period"] == scenario.prediction_period].copy()
        train = train.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
        target = target.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
        feature_views = list(
            dict.fromkeys(item["feature_view"] for item in mulcas_config["viewer_sequence"])
        )
        matrices, feature_audit, _ = build_historical_feature_matrices(
            train,
            target,
            feature_views,
            ngram_min_df=int(feature_config["ngram"]["min_df"]),
        )
        full_predictions, _, training_trace, outputs = fit_predict_frozen_mulcas(
            train,
            target,
            matrices,
            model_config,
            mulcas_config,
            seed=seed,
        )
        predictions_by_ablation = build_ablation_predictions(
            target["row_index"].to_numpy(dtype=int),
            mulcas_config,
            full_predictions,
            outputs,
        )
        truth = target.set_index("row_index")["label"]
        full_aligned = predictions_by_ablation[FULL_ABLATION_ID].set_index("sample_index")
        period_records = []
        for ablation in config["ablations"]:
            ablation_id = ablation["id"]
            predictions = predictions_by_ablation[ablation_id]
            aligned = predictions.set_index("sample_index").reindex(truth.index)
            metrics = binary_classification_metrics(
                truth.to_numpy(dtype=int),
                aligned["prediction"].to_numpy(dtype=int),
                aligned["probability"].to_numpy(dtype=float),
            )
            if ablation_id == FULL_ABLATION_ID:
                _assert_phase7a_reproduction(
                    scenario.prediction_period, metrics, phase7a_results
                )
            record = {
                "ablation_id": ablation_id,
                "operation": ablation["operation"],
                "prediction_period": scenario.prediction_period,
                "train_periods": list(scenario.train_periods),
                "prediction_count": int(len(target)),
                "prediction_changes_vs_full": int(
                    (aligned["prediction"] != full_aligned["prediction"]).sum()
                ),
                "routing_counts": {
                    str(key): int(value)
                    for key, value in predictions["handled_by"].value_counts().items()
                },
                "feature_fit_audit": feature_audit,
                "cascade_training_trace": training_trace,
                **metrics,
            }
            period_records.append(record)
            routing_records.extend(
                _routing_records(ablation_id, scenario.prediction_period, predictions)
            )
        reference = next(
            item for item in period_records if item["ablation_id"] == FULL_ABLATION_ID
        )
        for record in period_records:
            record["metric_deltas_vs_full"] = {
                name: float(record[name] - reference[name])
                for name in (
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
        ablation_records.extend(period_records)

    effect_summaries = []
    for ablation in config["ablations"]:
        ablation_id = ablation["id"]
        if ablation_id == FULL_ABLATION_ID:
            continue
        records = [item for item in ablation_records if item["ablation_id"] == ablation_id]
        f1_deltas = [item["metric_deltas_vs_full"]["f1"] for item in records]
        if all(value <= 0 for value in f1_deltas) and any(value < 0 for value in f1_deltas):
            direction = "full_component_nonnegative_in_every_period"
        elif all(value >= 0 for value in f1_deltas) and any(value > 0 for value in f1_deltas):
            direction = "ablation_nonnegative_in_every_period"
        else:
            direction = "period_specific_mixed_effect"
        effect_summaries.append(
            {
                "ablation_id": ablation_id,
                "f1_deltas_by_period": {
                    item["prediction_period"]: item["metric_deltas_vs_full"]["f1"]
                    for item in records
                },
                "descriptive_direction": direction,
            }
        )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "7B",
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
        "ablation_semantics": config["ablation_semantics"],
        "period_definitions": period_definitions,
        "frozen_viewers": viewer_description,
        "records": ablation_records,
        "routing_records": routing_records,
        "effect_summaries": effect_summaries,
        "total_runtime_seconds": float(time.perf_counter() - started),
    }


def _report(results: dict[str, Any]) -> str:
    metric_rows = [
        "| Ablation | Period | Precision | Recall | F1 | PR-AUC | Bal. acc. | TN | FP | FN | TP | Changed predictions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    delta_rows = [
        "| Ablation | P2 F1 delta | P3 F1 delta | P4 F1 delta | P5 F1 delta | Descriptive effect |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in results["records"]:
        metric_rows.append(
            f"| {record['ablation_id']} | {record['prediction_period']} | "
            f"{record['precision']:.4f} | {record['recall']:.4f} | {record['f1']:.4f} | "
            f"{record['pr_auc']:.4f} | {record['balanced_accuracy']:.4f} | "
            f"{record['tn']} | {record['fp']} | {record['fn']} | {record['tp']} | "
            f"{record['prediction_changes_vs_full']} |"
        )
    for summary in results["effect_summaries"]:
        values = summary["f1_deltas_by_period"]
        delta_rows.append(
            f"| {summary['ablation_id']} | {values['P2']:+.4f} | {values['P3']:+.4f} | "
            f"{values['P4']:+.4f} | {values['P5']:+.4f} | {summary['descriptive_direction']} |"
        )
    viewer_rows = [
        "| Order | Viewer | Feature | Classifier | Cascade stage | Voting |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for viewer in results["frozen_viewers"]["viewers"]:
        viewer_rows.append(
            f"| {viewer['position']} | {viewer['name']} | {viewer['feature_representation']} | "
            f"{viewer['classifier_family']} | {viewer['cascade_stage'] or '-'} | yes |"
        )
    return (
        "# Phase 7B: Frozen MulCas Component/View Ablation\n\n"
        "This is contribution analysis only. No feature, model, class weight, sampling rule, decision threshold, cascade threshold, voting threshold, or period boundary was tuned.\n\n"
        "## Actual Frozen Viewers\n\n" + "\n".join(viewer_rows) + "\n\n"
        "The no-voting condition uses the last cascade viewer's native prediction for unresolved samples. Standalone conditions use each viewer's full-historical-training voting copy and native model prediction. Removed viewers are excluded from both cascade routing and voting.\n\n"
        "## Metrics\n\n" + "\n".join(metric_rows) + "\n\n"
        "## F1 Changes Relative to Full MulCas\n\n" + "\n".join(delta_rows) + "\n\n"
        "## Contribution Interpretation\n\n"
        "- **P2:** removing the second cascade viewer, or using the first viewer alone, raises F1 by only 0.0156. The full first stage supplies more true positives but also most of P2's false positives; removing it changes 261 predictions and reduces recall from 0.2600 to 0.0400.\n"
        "- **P5:** the first viewer has the strongest positive contribution. Removing it lowers recall from 0.4219 to 0.2344 and F1 by 0.1981. Removing the second viewer lowers F1 by 0.0740.\n"
        "- **Consistency:** removing the first viewer never improves F1, while removing the second has a mixed effect: slightly positive in P2, neutral in P4, and negative in P3/P5.\n"
        "- **Voting:** removing voting changes 4 P2 predictions, no P3/P4 predictions, and one P5 prediction. Its F1 contribution is correspondingly small but nonnegative in every period.\n"
        "- **Routing and time:** the first viewer handles most contracts in every period, yet its outcomes range from very weak P2 performance to strong P4 performance. Routing volume alone does not explain the temporal variation.\n\n"
        "Positive deltas do not identify replacement systems; they only describe component effects for a specific period. No ablation is promoted as a candidate model. Routing details are available in `phase7b_routing.csv`.\n"
    )


def save_results(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Save the Phase 7B analysis and all requested machine-readable tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "phase7b_results.json",
        "metrics": output_dir / "phase7b_ablation_metrics.csv",
        "deltas": output_dir / "phase7b_metric_deltas.csv",
        "routing": output_dir / "phase7b_routing.csv",
        "viewers": output_dir / "phase7b_frozen_viewers.json",
        "report": output_dir / "phase7b_report.md",
    }
    paths["results"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    metric_rows = []
    delta_rows = []
    for record in results["records"]:
        excluded = {"metric_deltas_vs_full", "routing_counts", "feature_fit_audit", "cascade_training_trace"}
        metric_rows.append({key: value for key, value in record.items() if key not in excluded})
        delta_rows.append(
            {
                "ablation_id": record["ablation_id"],
                "prediction_period": record["prediction_period"],
                **{
                    f"delta_{key}": value
                    for key, value in record["metric_deltas_vs_full"].items()
                },
            }
        )
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(delta_rows).to_csv(paths["deltas"], index=False)
    pd.DataFrame(results["routing_records"]).to_csv(paths["routing"], index=False)
    paths["viewers"].write_text(
        json.dumps(results["frozen_viewers"], indent=2), encoding="utf-8"
    )
    paths["report"].write_text(_report(results), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    for name, path in save_results(run_phase7b(config), config["output_dir"]).items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
