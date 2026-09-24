"""Ponzi-occurrence-based chronological split for the paper-aligned protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.load import DEFAULT_RAW_DATASET_PATH, read_dataset_columns
from src.data.split import (
    SplitConfig,
    assert_temporal_split_boundaries,
    summarize_split_counts,
)
from src.data.validate import require_columns
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "paper_aligned_split.yaml"


@dataclass(frozen=True)
class PonziOccurrenceSplitConfig:
    dataset: Path = DEFAULT_RAW_DATASET_PATH
    output_dir: Path = REPO_ROOT / "results" / "baseline" / "paper_aligned"
    ordering_column: str = "createdBlockNumber"
    address_column: str = "address"
    label_column: str = "label"
    positive_label: int = 1
    train_positive_end: int = 200
    validation_positive_end: int = 250
    expected_total_positives: int = 314


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> PonziOccurrenceSplitConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PonziOccurrenceSplitConfig(
        dataset=_resolve(raw["dataset"]),
        output_dir=_resolve(raw["output_dir"]),
        ordering_column=raw.get("ordering_column", "createdBlockNumber"),
        address_column=raw.get("address_column", "address"),
        label_column=raw.get("label_column", "label"),
        positive_label=int(raw.get("positive_label", 1)),
        train_positive_end=int(raw["positive_boundaries"]["train"]),
        validation_positive_end=int(raw["positive_boundaries"]["validation"]),
        expected_total_positives=int(raw["positive_boundaries"]["test"]),
    )


def assign_ponzi_occurrence_splits(
    frame: pd.DataFrame,
    config: PonziOccurrenceSplitConfig = PonziOccurrenceSplitConfig(),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assign whole creation blocks using the configured positive occurrences."""
    required = [config.address_column, config.ordering_column, config.label_column]
    require_columns(frame.columns, required)
    data = frame.loc[:, required].copy()
    data["row_index"] = frame.index.astype(int)
    data[config.ordering_column] = pd.to_numeric(
        data[config.ordering_column], errors="raise"
    )
    data = data.sort_values(
        [config.ordering_column, "row_index"], kind="mergesort"
    ).reset_index(drop=True)
    positives = data[data[config.label_column] == config.positive_label]
    if len(positives) != config.expected_total_positives:
        raise ValueError(
            f"Expected {config.expected_total_positives} positives, found {len(positives)}."
        )
    if not 0 < config.train_positive_end < config.validation_positive_end < len(positives):
        raise ValueError("Positive occurrence boundaries must be strictly increasing.")

    train_block = int(
        positives.iloc[config.train_positive_end - 1][config.ordering_column]
    )
    validation_block = int(
        positives.iloc[config.validation_positive_end - 1][config.ordering_column]
    )
    result = data.copy()
    result["split"] = "test"
    result.loc[result[config.ordering_column] <= train_block, "split"] = "train"
    result.loc[
        (result[config.ordering_column] > train_block)
        & (result[config.ordering_column] <= validation_block),
        "split",
    ] = "validation"
    assert_temporal_split_boundaries(result, config.ordering_column)
    if result["row_index"].duplicated().any() or len(result) != len(frame):
        raise AssertionError("Split membership must be complete and disjoint.")
    return result, {
        "train_end_block": train_block,
        "validation_end_block": validation_block,
    }


def build_paper_aligned_split(
    config: PonziOccurrenceSplitConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_dataset_columns(
        [config.address_column, config.ordering_column, config.label_column],
        path=config.dataset,
    )
    membership, boundaries = assign_ponzi_occurrence_splits(frame, config)
    summary_config = SplitConfig(
        ordering_column=config.ordering_column,
        address_column=config.address_column,
        label_column=config.label_column,
        positive_label=config.positive_label,
    )
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "ponzi_occurrence_chronological_whole_block_boundaries",
        "dataset_path": str(config.dataset),
        "positive_occurrence_boundaries": {
            "train_through": config.train_positive_end,
            "validation_through": config.validation_positive_end,
            "test_through": config.expected_total_positives,
        },
        "block_boundaries": boundaries,
        "row_count": int(len(membership)),
        "split_counts": summarize_split_counts(membership, summary_config),
        "assertions": {
            "chronology_preserved": True,
            "splits_complete_and_disjoint": True,
            "never_shuffled": True,
        },
    }
    return membership, metadata


def save_outputs(
    membership: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = output_dir / "split_membership.csv"
    metadata_path = output_dir / "split_metadata.json"
    membership.loc[
        :, ["row_index", "address", "createdBlockNumber", "label", "split"]
    ].to_csv(membership_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"membership": membership_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-aligned chronological splits.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = save_outputs(*build_paper_aligned_split(config), config.output_dir)
    print(f"Wrote paper-aligned split membership to {paths['membership']}")
    print(f"Wrote paper-aligned split metadata to {paths['metadata']}")


if __name__ == "__main__":
    main()
