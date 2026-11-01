"""Generate reproducible summaries for the published static feature views."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import spmatrix

from src.data.load import DEFAULT_RAW_DATASET_PATH, read_dataset_columns
from src.features.developer import ChronologicalCreatorTransformer
from src.features.opcode import (
    fit_training_transformer,
    make_opcode_ngram_count_vectorizer,
    make_opcode_ngram_tfidf_vectorizer,
    make_opcode_term_count_vectorizer,
    make_opcode_tfidf_vectorizer,
    transform_documents,
)
from src.features.word2vec import OpcodeWord2VecTransformer
from src.utils.paths import REPO_ROOT


DEFAULT_FEATURE_CONFIG_PATH = REPO_ROOT / "configs" / "feature_reproduction.yaml"


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_feature_config(path: Path = DEFAULT_FEATURE_CONFIG_PATH) -> dict[str, Any]:
    """Load and resolve Phase 3 feature configuration."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset", "split_membership", "summary_output", "artifact_dir"):
        config[key] = _resolve_repo_path(config[key])
    return config


def load_feature_source(
    dataset_path: Path,
    split_membership_path: Path,
) -> pd.DataFrame:
    """Load only fields required to reproduce feature views and attach splits."""
    source = read_dataset_columns(
        ["address", "opcode", "creator", "createdBlockNumber", "label"],
        path=dataset_path,
    )
    source["row_index"] = source.index.astype(int)
    membership = pd.read_csv(split_membership_path)
    joined = membership.loc[:, ["row_index", "split"]].merge(
        source,
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    if joined.isna().any().any():
        raise ValueError("Feature source contains missing values after split join.")
    return joined


def _sparse_summary(matrix: spmatrix) -> dict[str, Any]:
    rows, columns = matrix.shape
    total = rows * columns
    nonzero_per_row = matrix.getnnz(axis=1)
    return {
        "rows": int(rows),
        "dimensions": int(columns),
        "nonzero_values": int(matrix.nnz),
        "density": 0.0 if total == 0 else float(matrix.nnz / total),
        "sparsity": 1.0 if total == 0 else float(1.0 - matrix.nnz / total),
        "nonzero_per_contract": {
            "min": int(nonzero_per_row.min()),
            "mean": float(nonzero_per_row.mean()),
            "max": int(nonzero_per_row.max()),
        },
    }


def _dense_summary(matrix: np.ndarray) -> dict[str, Any]:
    rows, columns = matrix.shape
    total = matrix.size
    nonzero = int(np.count_nonzero(matrix))
    row_norms = np.linalg.norm(matrix, axis=1)
    return {
        "rows": int(rows),
        "dimensions": int(columns),
        "nonzero_values": nonzero,
        "density": 0.0 if total == 0 else float(nonzero / total),
        "sparsity": 1.0 if total == 0 else float(1.0 - nonzero / total),
        "row_l2_norm": {
            "min": float(row_norms.min()),
            "mean": float(row_norms.mean()),
            "max": float(row_norms.max()),
        },
        "zero_vector_contracts": int((row_norms == 0).sum()),
    }


def _creator_summary(matrix: np.ndarray) -> dict[str, Any]:
    positives = int(matrix[:, 0].sum())
    return {
        "rows": int(matrix.shape[0]),
        "dimensions": 1,
        "known_prior_positive_creator_count": positives,
        "unknown_creator_count": int(matrix.shape[0] - positives),
        "known_prior_positive_creator_share": float(positives / len(matrix)),
    }


def reproduce_feature_views(config: dict[str, Any]) -> dict[str, Any]:
    """Fit on train only and summarize transforms for all chronological splits."""
    frame = load_feature_source(config["dataset"], config["split_membership"])
    partitions = {
        name: frame.loc[frame["split"] == name].copy()
        for name in ("train", "validation", "test")
    }
    train_documents = partitions["train"]["opcode"].astype(str).tolist()

    artifact_dir: Path = config["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_summaries: dict[str, Any] = {}

    vectorizers = {
        "opcode_term_count": make_opcode_term_count_vectorizer(),
        "opcode_tfidf": make_opcode_tfidf_vectorizer(),
        "opcode_ngram_count_1_3": make_opcode_ngram_count_vectorizer(),
        "opcode_ngram_tfidf_1_3": make_opcode_ngram_tfidf_vectorizer(),
    }
    for view_name, vectorizer in vectorizers.items():
        fit_training_transformer(vectorizer, train_documents)
        joblib.dump(vectorizer, artifact_dir / f"{view_name}.joblib", compress=3)
        feature_summaries[view_name] = {
            "fitted_on": "train",
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "splits": {},
        }
        for split_name, partition in partitions.items():
            matrix = transform_documents(
                vectorizer,
                partition["opcode"].astype(str).tolist(),
            )
            feature_summaries[view_name]["splits"][split_name] = _sparse_summary(matrix)
            del matrix

    word2vec_config = config["word2vec"]
    word2vec = OpcodeWord2VecTransformer(
        vector_size=int(word2vec_config["vector_size"]),
        window=int(word2vec_config["window"]),
        min_count=int(word2vec_config["min_count"]),
        epochs=int(word2vec_config["epochs"]),
        seed=int(word2vec_config["seed"]),
        workers=int(word2vec_config["workers"]),
    ).fit(train_documents)
    joblib.dump(word2vec, artifact_dir / "opcode_word2vec.joblib", compress=3)
    feature_summaries["opcode_word2vec_mean"] = {
        "fitted_on": "train",
        "vocabulary_size": word2vec.vocabulary_size_,
        "splits": {},
    }
    for split_name, partition in partitions.items():
        matrix = word2vec.transform(partition["opcode"].astype(str).tolist())
        feature_summaries["opcode_word2vec_mean"]["splits"][split_name] = _dense_summary(
            matrix
        )
        del matrix

    creator = ChronologicalCreatorTransformer()
    train_creator = creator.fit_transform(partitions["train"])
    joblib.dump(creator, artifact_dir / "chronological_creator.joblib", compress=3)
    feature_summaries["chronological_creator"] = {
        "fitted_on": "train_prequential_strictly_earlier_blocks",
        "validation_test_state": "frozen_training_state",
        "positive_creators_in_training_state": len(creator.positive_creators_),
        "splits": {"train": _creator_summary(train_creator)},
    }
    for split_name in ("validation", "test"):
        matrix = creator.transform(
            partitions[split_name].loc[:, ["creator", "createdBlockNumber"]]
        )
        feature_summaries["chronological_creator"]["splits"][split_name] = (
            _creator_summary(matrix)
        )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(config["dataset"]),
        "split_membership_path": str(config["split_membership"]),
        "anti_leakage_policy": {
            "text_fit_partition": "train only",
            "validation_test_operation": "transform only",
            "creator_training": "strictly earlier blocks only",
            "creator_validation_test": "frozen training-label state; labels not read",
        },
        "feature_views": feature_summaries,
    }


def save_feature_summary(summary: dict[str, Any], output_path: Path) -> Path:
    """Save compact machine-readable feature-view measurements."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce published feature views.")
    parser.add_argument("--config", type=Path, default=DEFAULT_FEATURE_CONFIG_PATH)
    args = parser.parse_args()

    config = load_feature_config(args.config)
    output_path = save_feature_summary(
        reproduce_feature_views(config),
        config["summary_output"],
    )
    print(f"Wrote feature-view summary to {output_path}")
    print(f"Wrote fitted transformers to {config['artifact_dir']}")


if __name__ == "__main__":
    main()
