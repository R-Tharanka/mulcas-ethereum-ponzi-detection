"""Train, run, and report the frozen independent MulCas reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack, issparse
from scipy.special import expit

from src.evaluation.metrics import binary_classification_metrics
from src.features.developer import ChronologicalCreatorTransformer
from src.features.reproduce import load_feature_config, load_feature_source
from src.models.baselines import build_baseline_model
from src.models.mulcas import MulCasCascade, ViewerPrediction, ViewerSpec
from src.models.run_baselines import (
    TEXT_FEATURE_ARTIFACTS,
    load_baseline_config,
    validate_evaluation_protocol,
)
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_MULCAS_CONFIG_PATH = REPO_ROOT / "configs" / "mulcas_reproduction.yaml"


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_mulcas_config(path: Path = DEFAULT_MULCAS_CONFIG_PATH) -> dict[str, Any]:
    """Load and resolve the exact frozen MulCas reproduction configuration."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "baseline_config",
        "results_json",
        "trace_dir",
        "artifact_dir",
        "report_path",
    ):
        config[key] = _resolve_repo_path(config[key])
    if "existing_reconstruction_results" in config:
        config["existing_reconstruction_results"] = _resolve_repo_path(
            config["existing_reconstruction_results"]
        )
    for key in ("split_metadata", "feature_summary", "individual_baseline_results"):
        if key in config:
            config[key] = _resolve_repo_path(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def positive_probability(model: Any, features: Any) -> np.ndarray:
    """Return probabilities or deterministic sigmoid-mapped decision margins."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features), dtype=float)[:, 1]
    if hasattr(model, "decision_function"):
        margins = np.asarray(model.decision_function(features), dtype=float).reshape(-1)
        return expit(margins)
    raise TypeError("Viewer model must expose predict_proba or decision_function.")


def _partition_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split_name: frame.loc[frame["split"] == split_name].copy()
        for split_name in ("train", "validation", "test")
    }


def _append_developer_feature(features: Any, developer: np.ndarray) -> Any:
    if issparse(features):
        return hstack([features, developer], format="csr")
    return np.column_stack([np.asarray(features), developer])


def build_mulcas_feature_matrices(
    config: dict[str, Any],
    partitions: dict[str, pd.DataFrame],
    feature_artifact_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Build configured views with the leakage-safe developer feature appended."""
    developer = ChronologicalCreatorTransformer()
    developer_matrices = {
        "train": developer.fit_transform(partitions["train"]),
        "validation": developer.transform(
            partitions["validation"].loc[:, ["creator", "createdBlockNumber"]]
        ),
        "test": developer.transform(
            partitions["test"].loc[:, ["creator", "createdBlockNumber"]]
        ),
    }

    matrices: dict[str, dict[str, Any]] = {}
    feature_views = list(
        dict.fromkeys(viewer["feature_view"] for viewer in config["viewer_sequence"])
    )
    for feature_view in feature_views:
        artifact_mapping = config.get("feature_artifacts", TEXT_FEATURE_ARTIFACTS)
        transformer_path = feature_artifact_dir / artifact_mapping[feature_view]
        if not transformer_path.is_file():
            raise FileNotFoundError(
                f"Missing feature transformer: {transformer_path}. "
                "Run python -m src.features.reproduce first."
            )
        transformer = joblib.load(transformer_path)
        matrices[feature_view] = {}
        for split_name, partition in partitions.items():
            static_features = transformer.transform(
                partition["opcode"].astype(str).tolist()
            )
            matrices[feature_view][split_name] = _append_developer_feature(
                static_features,
                developer_matrices[split_name],
            )
    return matrices


def _viewer_prediction(
    model: Any,
    features: Any,
    sample_indices: np.ndarray,
) -> ViewerPrediction:
    return ViewerPrediction(
        probability=pd.Series(positive_probability(model, features), index=sample_indices),
        prediction=pd.Series(
            np.asarray(model.predict(features), dtype=int),
            index=sample_indices,
        ),
    )


def train_mulcas_viewers(
    config: dict[str, Any],
    baseline_config: dict[str, Any],
    partitions: dict[str, pd.DataFrame],
    matrices: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, ViewerPrediction]],
    dict[str, dict[str, ViewerPrediction]],
    list[dict[str, Any]],
]:
    """Train residual cascade viewers and full-training voting viewers."""
    seed = int(config["seed"])
    artifact_dir: Path = config["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    labels = partitions["train"]["label"].to_numpy(dtype=int)
    split_indices = {
        name: partition["row_index"].to_numpy()
        for name, partition in partitions.items()
    }
    cascade_outputs = {"validation": {}, "test": {}}
    voting_outputs = {"validation": {}, "test": {}}
    training_trace: list[dict[str, Any]] = []

    residual_positions = np.arange(len(partitions["train"]), dtype=int)
    cascade_viewers = config["viewer_sequence"][: int(config["complexity_control_n"])]
    for stage, viewer in enumerate(cascade_viewers, start=1):
        feature_view = viewer["feature_view"]
        model = build_baseline_model(
            viewer["model"],
            baseline_config["models"][viewer["model"]],
            seed,
        )
        train_features = matrices[feature_view]["train"][residual_positions]
        train_labels = labels[residual_positions]
        model.fit(train_features, train_labels)
        joblib.dump(
            model,
            artifact_dir / f"cascade_stage_{stage}__{viewer['name']}.joblib",
            compress=3,
        )

        probabilities = positive_probability(model, train_features)
        confident = (probabilities < float(viewer["negative_threshold"])) | (
            probabilities > float(viewer["positive_threshold"])
        )
        training_trace.append(
            {
                "stage": stage,
                "viewer": viewer["name"],
                "training_rows_in": int(len(residual_positions)),
                "confident_training_rows": int(confident.sum()),
                "training_rows_out": int((~confident).sum()),
                "positive_training_rows_in": int(train_labels.sum()),
                "positive_training_rows_out": int(train_labels[~confident].sum()),
            }
        )
        residual_positions = residual_positions[~confident]

        for split_name in ("validation", "test"):
            cascade_outputs[split_name][viewer["name"]] = _viewer_prediction(
                model,
                matrices[feature_view][split_name],
                split_indices[split_name],
            )

    for viewer in config["viewer_sequence"]:
        feature_view = viewer["feature_view"]
        model = build_baseline_model(
            viewer["model"],
            baseline_config["models"][viewer["model"]],
            seed,
        )
        model.fit(matrices[feature_view]["train"], labels)
        joblib.dump(
            model,
            artifact_dir / f"voting_full_train__{viewer['name']}.joblib",
            compress=3,
        )
        for split_name in ("validation", "test"):
            voting_outputs[split_name][viewer["name"]] = _viewer_prediction(
                model,
                matrices[feature_view][split_name],
                split_indices[split_name],
            )

    return cascade_outputs, voting_outputs, training_trace


def _comparison_to_published(
    experimental_metrics: dict[str, Any],
    published: dict[str, Any],
) -> dict[str, Any]:
    return {
        metric: {
            "published": float(published[metric]),
            "experimental": float(experimental_metrics[metric]),
            "difference_experimental_minus_published": float(
                experimental_metrics[metric] - published[metric]
            ),
        }
        for metric in ("precision", "recall", "f1")
    }


def run_mulcas_reproduction(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Train and evaluate the paper-aligned frozen reconstruction."""
    validate_evaluation_protocol(config)
    set_random_seed(int(config["seed"]))
    baseline_config = load_baseline_config(config["baseline_config"])
    validate_evaluation_protocol(baseline_config)
    feature_config = load_feature_config(baseline_config["feature_config"])
    frame = load_feature_source(
        feature_config["dataset"],
        feature_config["split_membership"],
    )
    partitions = _partition_frame(frame)
    matrices = build_mulcas_feature_matrices(
        config,
        partitions,
        feature_config["artifact_dir"],
    )
    cascade_outputs, voting_outputs, training_trace = train_mulcas_viewers(
        config,
        baseline_config,
        partitions,
        matrices,
    )

    cascade_viewer_configs = config["viewer_sequence"][
        : int(config["complexity_control_n"])
    ]
    cascade = MulCasCascade(
        cascade_viewers=[
            ViewerSpec(
                name=viewer["name"],
                negative_threshold=float(viewer["negative_threshold"]),
                positive_threshold=float(viewer["positive_threshold"]),
            )
            for viewer in cascade_viewer_configs
        ],
        voting_viewers=[viewer["name"] for viewer in config["viewer_sequence"]],
        voting_threshold=float(config["voting_threshold"]),
    )

    split_results: dict[str, Any] = {}
    traces: dict[str, pd.DataFrame] = {}
    for split_name in ("validation", "test"):
        partition = partitions[split_name]
        sample_indices = partition["row_index"].tolist()
        cascade_result = cascade.predict(
            sample_indices,
            cascade_outputs[split_name],
            voting_outputs[split_name],
        )
        truth = partition.set_index("row_index")["label"].reindex(sample_indices)
        predictions = cascade_result.predictions.set_index("sample_index")
        metrics = binary_classification_metrics(
            truth.to_numpy(dtype=int),
            predictions["prediction"].to_numpy(dtype=int),
            predictions["probability"].to_numpy(dtype=float),
        )
        split_results[split_name] = {
            **metrics,
            "routing_counts": {
                key: int(value)
                for key, value in predictions["handled_by"].value_counts().items()
            },
        }

        trace = cascade_result.traces.copy()
        address_lookup = partition.set_index("row_index")["address"]
        trace["address"] = trace["sample_index"].map(address_lookup)
        trace = trace.merge(
            cascade_result.predictions.rename(
                columns={
                    "prediction": "final_prediction",
                    "probability": "final_probability",
                }
            ),
            on="sample_index",
            how="left",
            validate="many_to_one",
        )
        trace["split"] = split_name
        traces[split_name] = trace

    published = config["published_reference"]
    config_snapshot = yaml.safe_load(config["config_path"].read_text(encoding="utf-8"))
    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "baseline_status": "frozen" if config["frozen_baseline"] else "draft",
        "configuration": {
            "path": str(config["config_path"]),
            "sha256": config["config_sha256"],
            "snapshot": config_snapshot,
        },
        "methodology": {
            "cascade_training": "Each of N=2 viewers fitted on residual training rows.",
            "voting_training": "All voting viewers fitted on the full training partition.",
            "developer_feature": "Leakage-safe creator flag concatenated to every view.",
            "validation_policy": "Validation labels used for reporting only in Phase 5.",
            "test_policy": "Final-test labels used only after predictions for reporting.",
            "score_mapping": config["score_mapping"],
            "reconstruction_note": config.get(
                "reconstruction_note",
                "Algorithm and N/theta/lambda follow the paper. Existing 70/15/15 "
                "chronological splits and independently reproduced features/classifiers "
                "differ from the paper's experimental environment.",
            ),
        },
        "cascade_training_trace": training_trace,
        "published_mulcas": {
            "label": "Published MulCas (paper reference; not generated here)",
            "precision": float(published["precision"]),
            "recall": float(published["recall"]),
            "f1": float(published["f1"]),
            "accuracy": None,
            "pr_auc": None,
            "balanced_accuracy": None,
            "confusion_matrix": None,
        },
        "experimental_reconstruction": {
            "label": config.get(
                "experimental_label", "Our independent MulCas reconstruction"
            ),
            "validation": split_results["validation"],
            "test": split_results["test"],
        },
        "test_differences_from_published": _comparison_to_published(
            split_results["test"],
            published,
        ),
    }
    previous_path = config.get("existing_reconstruction_results")
    if previous_path is not None:
        previous = json.loads(Path(previous_path).read_text(encoding="utf-8"))
        result["existing_70_15_15_reconstruction"] = {
            "label": "Existing 70/15/15 independent reconstruction",
            "source_path": str(previous_path),
            "test": previous["experimental_reconstruction"]["test"],
        }
    return result, traces


def _paper_aligned_protocol_section(config: dict[str, Any]) -> str:
    required = ("split_metadata", "feature_summary", "individual_baseline_results")
    if not all(key in config for key in required):
        return ""
    split_metadata = json.loads(config["split_metadata"].read_text(encoding="utf-8"))
    feature_summary = json.loads(config["feature_summary"].read_text(encoding="utf-8"))
    baseline_results = json.loads(
        config["individual_baseline_results"].read_text(encoding="utf-8")
    )
    split_lines = []
    for split_name in ("train", "validation", "test"):
        values = split_metadata["split_counts"][split_name]
        split_lines.append(
            f"| {split_name} | {values['rows']} | {values['positive_count']} | "
            f"{values['negative_count']} | {values['negative_to_positive_ratio']:.6f} | "
            f"{values['block_min']} | {values['block_max']} |"
        )
    dimension_lines = [
        f"| {name} | {dimension} |"
        for name, dimension in feature_summary[
            "dimensions_after_creator_concatenation"
        ].items()
    ]
    validation_records = [
        record
        for record in baseline_results["records"]
        if record["split"] == "validation"
    ]
    selected_lines = []
    for family in ("ridge", "svm", "xgboost"):
        selected = max(
            (record for record in validation_records if record["model"] == family),
            key=lambda record: record["f1"],
        )
        test_record = next(
            record
            for record in baseline_results["records"]
            if record["split"] == "test"
            and record["model"] == family
            and record["feature_view"] == selected["feature_view"]
        )
        selected_lines.append(
            f"| {family} | {selected['feature_view']} | {selected['f1']:.6f} | "
            f"{test_record['precision']:.6f} | {test_record['recall']:.6f} | "
            f"{test_record['f1']:.6f} | {test_record['pr_auc']:.6f} |"
        )
    return f"""
## Corrected Partition Statistics

| Split | Rows | Positive | Negative | Negative:positive | Block min | Block max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(split_lines)}

## Feature Dimensions

Dimensions include the appended chronological creator feature.

| Feature view | Dimensions |
| --- | ---: |
{chr(10).join(dimension_lines)}

## Individual Baseline Summary

One feature view per model family is selected by validation F1 only. The complete
fixed matrix remains in `{config['individual_baseline_results'].as_posix()}`.

| Model | Validation-selected feature | Validation F1 | Test precision | Test recall | Test F1 | Test PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(selected_lines)}
"""


def save_mulcas_outputs(
    result: dict[str, Any],
    traces: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> None:
    """Save machine-readable result, detailed traces, and Markdown report."""
    results_path: Path = config["results_json"]
    trace_dir: Path = config["trace_dir"]
    report_path: Path = config["report_path"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for split_name, trace in traces.items():
        trace.to_csv(trace_dir / f"{split_name}_trace.csv", index=False)

    test = result["experimental_reconstruction"]["test"]
    published = result["published_mulcas"]
    differences = result["test_differences_from_published"]
    previous_section = ""
    if "existing_70_15_15_reconstruction" in result:
        previous = result["existing_70_15_15_reconstruction"]["test"]
        previous_section = f"""
## Three-Way Result Distinction

| Metric | Published paper | Existing 70/15/15 | Corrected paper-aligned |
| --- | ---: | ---: | ---: |
| Precision | {published['precision']:.3f} | {previous['precision']:.6f} | {test['precision']:.6f} |
| Recall | {published['recall']:.3f} | {previous['recall']:.6f} | {test['recall']:.6f} |
| F1 | {published['f1']:.3f} | {previous['f1']:.6f} | {test['f1']:.6f} |

The paper also states F1 = 0.743 in nearby prose; this conflicts with Table 5
and is retained as an unresolved source inconsistency.
"""
    protocol_section = _paper_aligned_protocol_section(config)
    report = f"""# Frozen MulCas Reproduction Baseline

This report distinguishes published reference values from metrics generated by
our independent reconstruction. The exact configuration and SHA-256 are embedded
in `{results_path.as_posix()}`.

## Configuration

- Status: `{result['baseline_status']}`
- Cascade complexity N: `{config['complexity_control_n']}`
- Voting threshold lambda: `{config['voting_threshold']}`
- Voting viewer count: `{len(config['viewer_sequence'])}`
- Configuration SHA-256: `{result['configuration']['sha256']}`

Algorithm 2 and the reported N/theta/lambda settings follow the paper. Our
chronological split, reproduced features, software versions, and independently
trained classifiers differ from the original experimental environment.

## Final-Test Metrics

| Metric | Published MulCas | Independent reconstruction | Difference |
| --- | ---: | ---: | ---: |
| Accuracy | Not reported | {test['accuracy']:.6f} | Not comparable |
| Balanced accuracy | Not reported | {test['balanced_accuracy']:.6f} | Not comparable |
| Precision | {published['precision']:.3f} | {test['precision']:.6f} | {differences['precision']['difference_experimental_minus_published']:+.6f} |
| Recall | {published['recall']:.3f} | {test['recall']:.6f} | {differences['recall']['difference_experimental_minus_published']:+.6f} |
| F1 | {published['f1']:.3f} | {test['f1']:.6f} | {differences['f1']['difference_experimental_minus_published']:+.6f} |
| PR-AUC | Not reported | {test['pr_auc']:.6f} | Not comparable |

## Confusion Matrix

| | Predicted negative | Predicted positive |
| --- | ---: | ---: |
| Actual negative | TN = {test['tn']} | FP = {test['fp']} |
| Actual positive | FN = {test['fn']} | TP = {test['tp']} |

## Routing

```json
{json.dumps(test['routing_counts'], indent=2)}
```

Detailed validation and final-test traces are stored under
`{trace_dir.as_posix()}`.

{previous_section}

{protocol_section}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen MulCas reconstruction.")
    parser.add_argument("--config", type=Path, default=DEFAULT_MULCAS_CONFIG_PATH)
    args = parser.parse_args()
    config = load_mulcas_config(args.config)
    result, traces = run_mulcas_reproduction(config)
    save_mulcas_outputs(result, traces, config)
    print(f"Wrote MulCas results to {config['results_json']}")
    print(f"Wrote traces to {config['trace_dir']}")
    print(f"Wrote frozen baseline report to {config['report_path']}")


if __name__ == "__main__":
    main()
