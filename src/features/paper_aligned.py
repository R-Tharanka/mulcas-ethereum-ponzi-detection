"""Paper-aligned feature variants for the frozen Phase 5.5 protocol."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import yaml

from src.features.developer import ChronologicalCreatorTransformer
from src.features.opcode import (
    fit_training_transformer,
    make_opcode_term_count_vectorizer,
    make_opcode_tfidf_vectorizer,
    make_paper_aligned_ngram_count_vectorizer,
    make_paper_aligned_ngram_tfidf_vectorizer,
    transform_documents,
)
from src.features.reproduce import (
    _creator_summary,
    _dense_summary,
    _sparse_summary,
    load_feature_source,
)
from src.features.word2vec import (
    IDFWeightedOpcodeWord2VecTransformer,
    OpcodeWord2VecTransformer,
)
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "paper_aligned_features.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset", "split_membership", "summary_output", "artifact_dir"):
        config[key] = _resolve(config[key])
    return config


def _word2vec_from_config(config: dict[str, Any], weighted: bool) -> Any:
    constructor = (
        IDFWeightedOpcodeWord2VecTransformer if weighted else OpcodeWord2VecTransformer
    )
    values = config["word2vec"]
    return constructor(
        vector_size=int(values["vector_size"]),
        window=int(values["window"]),
        min_count=int(values["min_count"]),
        epochs=int(values["epochs"]),
        seed=int(values["seed"]),
        workers=int(values["workers"]),
    )


def reproduce_paper_aligned_features(config: dict[str, Any]) -> dict[str, Any]:
    """Fit all paper-aligned transformers on training data only."""
    frame = load_feature_source(config["dataset"], config["split_membership"])
    partitions = {
        name: frame.loc[frame["split"] == name].copy()
        for name in ("train", "validation", "test")
    }
    train_documents = partitions["train"]["opcode"].astype(str).tolist()
    artifact_dir: Path = config["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    min_df = int(config["ngram"]["min_df"])
    summaries: dict[str, Any] = {}

    vectorizers = {
        "opcode_term_count": make_opcode_term_count_vectorizer(),
        "opcode_tfidf": make_opcode_tfidf_vectorizer(),
        "opcode_ngram_count_1_3_df_filtered": (
            make_paper_aligned_ngram_count_vectorizer(min_df)
        ),
        "opcode_ngram_tfidf_1_3_df_filtered": (
            make_paper_aligned_ngram_tfidf_vectorizer(min_df)
        ),
    }
    for name, transformer in vectorizers.items():
        fit_training_transformer(transformer, train_documents)
        joblib.dump(transformer, artifact_dir / f"{name}.joblib", compress=3)
        summaries[name] = {
            "fitted_on": "train",
            "vocabulary_size": int(len(transformer.vocabulary_)),
            "document_frequency_minimum": min_df if "ngram" in name else 1,
            "splits": {},
        }
        for split_name, partition in partitions.items():
            matrix = transform_documents(
                transformer, partition["opcode"].astype(str).tolist()
            )
            summaries[name]["splits"][split_name] = _sparse_summary(matrix)

    for name, weighted in (
        ("opcode_word2vec_mean", False),
        ("opcode_word2vec_idf_weighted", True),
    ):
        transformer = _word2vec_from_config(config, weighted).fit(train_documents)
        joblib.dump(transformer, artifact_dir / f"{name}.joblib", compress=3)
        summaries[name] = {
            "fitted_on": "train",
            "aggregation": "idf_weighted_sum" if weighted else "unweighted_mean",
            "vocabulary_size": transformer.vocabulary_size_,
            "splits": {},
        }
        for split_name, partition in partitions.items():
            matrix = transformer.transform(partition["opcode"].astype(str).tolist())
            summaries[name]["splits"][split_name] = _dense_summary(matrix)

    creator = ChronologicalCreatorTransformer()
    train_creator = creator.fit_transform(partitions["train"])
    joblib.dump(creator, artifact_dir / "chronological_creator.joblib", compress=3)
    creator_summary = {
        "fitted_on": "train_prequential_strictly_earlier_blocks",
        "use": "appended_to_every_static_opcode_view",
        "splits": {"train": _creator_summary(train_creator)},
    }
    for split_name in ("validation", "test"):
        values = creator.transform(
            partitions[split_name].loc[:, ["creator", "createdBlockNumber"]]
        )
        creator_summary["splits"][split_name] = _creator_summary(values)
    summaries["chronological_creator"] = creator_summary

    combined_dimensions = {
        name: int(details["splits"]["train"]["dimensions"] + 1)
        for name, details in summaries.items()
        if name != "chronological_creator"
    }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "paper_aligned_phase_5_5",
        "dataset_path": str(config["dataset"]),
        "split_membership_path": str(config["split_membership"]),
        "ngram_filter": config["ngram"],
        "anti_leakage_policy": {
            "all_learned_feature_state": "fit on training partition only",
            "validation_test": "transform only",
            "creator_training": "strictly earlier blocks",
            "creator_later_splits": "frozen training state without labels",
        },
        "feature_views": summaries,
        "dimensions_after_creator_concatenation": combined_dimensions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-aligned features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = reproduce_paper_aligned_features(config)
    config["summary_output"].parent.mkdir(parents=True, exist_ok=True)
    config["summary_output"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote paper-aligned feature summary to {config['summary_output']}")
    print(f"Wrote fitted transformers to {config['artifact_dir']}")


if __name__ == "__main__":
    main()
