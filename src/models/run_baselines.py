"""Run fixed individual baseline models across reproduced feature views."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack, issparse

from src.evaluation.metrics import binary_classification_metrics
from src.features.developer import ChronologicalCreatorTransformer
from src.features.reproduce import load_feature_config, load_feature_source
from src.models.baselines import build_baseline_model, positive_class_scores
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "baseline_models.yaml"

TEXT_FEATURE_ARTIFACTS: dict[str, str] = {
    "opcode_term_count": "opcode_term_count.joblib",
    "opcode_tfidf": "opcode_tfidf.joblib",
    "opcode_ngram_count_1_3": "opcode_ngram_count_1_3.joblib",
    "opcode_ngram_tfidf_1_3": "opcode_ngram_tfidf_1_3.joblib",
    "opcode_word2vec_mean": "opcode_word2vec.joblib",
}


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_baseline_config(path: Path = DEFAULT_BASELINE_CONFIG_PATH) -> dict[str, Any]:
    """Load fixed Phase 4 model and output configuration."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("feature_config", "results_json", "results_csv", "artifact_dir"):
        config[key] = _resolve_repo_path(config[key])
    return config


def validate_evaluation_protocol(config: dict[str, Any]) -> None:
    """Reject configurations that authorize final-test-driven selection."""
    if config.get("selection_split", "validation") != "validation":
        raise ValueError("Model selection must use the validation split only.")
    if config.get("final_test_usage", "reporting_only") != "reporting_only":
        raise ValueError("Final-test labels may be used for reporting only.")


def _partition_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split_name: frame.loc[frame["split"] == split_name].copy()
        for split_name in ("train", "validation", "test")
    }


def build_feature_matrices(
    feature_view: str,
    partitions: dict[str, pd.DataFrame],
    feature_artifact_dir: Path,
    feature_artifacts: dict[str, str] | None = None,
    append_creator: bool = False,
) -> dict[str, Any]:
    """Transform all splits using a training-fitted Phase 3 transformer."""
    if feature_view == "chronological_creator":
        transformer = ChronologicalCreatorTransformer()
        matrices = {"train": transformer.fit_transform(partitions["train"])}
        for split_name in ("validation", "test"):
            matrices[split_name] = transformer.transform(
                partitions[split_name].loc[:, ["creator", "createdBlockNumber"]]
            )
        return matrices

    artifact_mapping = feature_artifacts or TEXT_FEATURE_ARTIFACTS
    artifact_name = artifact_mapping[feature_view]
    artifact_path = feature_artifact_dir / artifact_name
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Missing fitted feature transformer: {artifact_path}. "
            "Run python -m src.features.reproduce first."
        )
    transformer = joblib.load(artifact_path)
    matrices = {
        split_name: transformer.transform(partition["opcode"].astype(str).tolist())
        for split_name, partition in partitions.items()
    }
    if append_creator:
        creator = ChronologicalCreatorTransformer()
        creator_values = {"train": creator.fit_transform(partitions["train"])}
        for split_name in ("validation", "test"):
            creator_values[split_name] = creator.transform(
                partitions[split_name].loc[:, ["creator", "createdBlockNumber"]]
            )
        matrices = {
            split_name: hstack([matrix, creator_values[split_name]], format="csr")
            if issparse(matrix)
            else np.column_stack([np.asarray(matrix), creator_values[split_name]])
            for split_name, matrix in matrices.items()
        }
    return matrices


def evaluate_model_splits(
    model: Any,
    matrices: dict[str, Any],
    labels: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Evaluate an already-fitted model without changing model state."""
    split_results: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        predictions = np.asarray(model.predict(matrices[split_name]), dtype=int)
        scores = positive_class_scores(model, matrices[split_name])
        split_results.append(
            {
                "split": split_name,
                **binary_classification_metrics(
                    labels[split_name],
                    predictions,
                    scores,
                ),
            }
        )
    return split_results


def run_individual_baselines(config: dict[str, Any]) -> dict[str, Any]:
    """Fit fixed baselines on train and evaluate unchanged on later splits."""
    validate_evaluation_protocol(config)
    seed = int(config["seed"])
    set_random_seed(seed)
    feature_config = load_feature_config(config["feature_config"])
    frame = load_feature_source(
        feature_config["dataset"],
        feature_config["split_membership"],
    )
    partitions = _partition_frame(frame)
    labels = {
        split_name: partition["label"].to_numpy(dtype=int)
        for split_name, partition in partitions.items()
    }

    artifact_dir: Path = config["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    configured_artifacts = config.get("feature_artifacts")
    feature_artifacts = configured_artifacts or TEXT_FEATURE_ARTIFACTS
    feature_views = config.get(
        "feature_views",
        [*TEXT_FEATURE_ARTIFACTS, "chronological_creator"],
    )
    append_creator = bool(config.get("append_creator_to_static_views", False))

    for feature_view in feature_views:
        matrices = build_feature_matrices(
            feature_view,
            partitions,
            feature_config["artifact_dir"],
            feature_artifacts,
            append_creator=append_creator and feature_view != "chronological_creator",
        )
        for model_family, parameters in config["models"].items():
            model = build_baseline_model(model_family, parameters, seed)
            fit_started = time.perf_counter()
            model.fit(matrices["train"], labels["train"])
            fit_seconds = time.perf_counter() - fit_started

            model_path = artifact_dir / f"{feature_view}__{model_family}.joblib"
            joblib.dump(model, model_path, compress=3)
            for split_metrics in evaluate_model_splits(model, matrices, labels):
                records.append(
                    {
                        "model": model_family,
                        "feature_view": feature_view,
                        "parameters": parameters,
                        "split": split_metrics.pop("split"),
                        "seed": seed,
                        "fit_partition": "train",
                        "fit_seconds": fit_seconds,
                        "decision_threshold": "estimator_default",
                        **split_metrics,
                    }
                )
        del matrices

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(feature_config["dataset"]),
        "split_membership_path": str(feature_config["split_membership"]),
        "seed": seed,
        "selection_policy": (
            "Fixed baseline parameters; no hyperparameter or threshold optimization. "
            "Final-test data is used only for reporting after fitting."
        ),
        "developer_feature_policy": "appended_to_static_views"
        if append_creator
        else "standalone_historical_view",
        "records": records,
    }


def save_baseline_results(
    results: dict[str, Any],
    json_path: Path,
    csv_path: Path,
) -> dict[str, Path]:
    """Save complete JSON and flat CSV baseline result records."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    flat_records = []
    for record in results["records"]:
        flat_record = dict(record)
        flat_record["parameters"] = json.dumps(record["parameters"], sort_keys=True)
        flat_records.append(flat_record)
    pd.DataFrame(flat_records).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed individual baselines.")
    parser.add_argument("--config", type=Path, default=DEFAULT_BASELINE_CONFIG_PATH)
    args = parser.parse_args()

    config = load_baseline_config(args.config)
    paths = save_baseline_results(
        run_individual_baselines(config),
        config["results_json"],
        config["results_csv"],
    )
    print(f"Wrote baseline JSON to {paths['json']}")
    print(f"Wrote baseline CSV to {paths['csv']}")
    print(f"Wrote fitted models to {config['artifact_dir']}")


if __name__ == "__main__":
    main()
