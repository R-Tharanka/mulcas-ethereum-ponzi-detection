"""Leakage-safe chronological train/validation/test splitting."""

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
from src.data.validate import is_chronologically_ordered, require_columns
from src.utils.paths import REPO_ROOT, RESULTS_DIR


DEFAULT_SPLIT_CONFIG_PATH = REPO_ROOT / "configs" / "data_split.yaml"
DEFAULT_SPLIT_OUTPUT_DIR = RESULTS_DIR / "baseline"


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic chronological splitting."""

    dataset: Path = DEFAULT_RAW_DATASET_PATH
    output_dir: Path = DEFAULT_SPLIT_OUTPUT_DIR
    ordering_column: str = "createdBlockNumber"
    address_column: str = "address"
    label_column: str = "label"
    positive_label: int = 1
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    duplicate_check_columns: tuple[str, ...] = (
        "address",
        "opcode",
        "code",
        "contractCode",
        "creationCode",
    )


def load_split_config(path: Path = DEFAULT_SPLIT_CONFIG_PATH) -> SplitConfig:
    """Load split configuration from YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    fractions = raw.get("split_fractions", {})
    return SplitConfig(
        dataset=REPO_ROOT / raw.get("dataset", str(DEFAULT_RAW_DATASET_PATH)),
        output_dir=REPO_ROOT / raw.get("output_dir", str(DEFAULT_SPLIT_OUTPUT_DIR)),
        ordering_column=raw.get("ordering_column", "createdBlockNumber"),
        address_column=raw.get("address_column", "address"),
        label_column=raw.get("label_column", "label"),
        positive_label=int(raw.get("positive_label", 1)),
        train_fraction=float(fractions.get("train", 0.70)),
        validation_fraction=float(fractions.get("validation", 0.15)),
        test_fraction=float(fractions.get("test", 0.15)),
        duplicate_check_columns=tuple(raw.get("duplicate_check_columns", ())),
    )


def validate_split_config(config: SplitConfig) -> None:
    """Validate split fraction settings."""
    fractions = [
        config.train_fraction,
        config.validation_fraction,
        config.test_fraction,
    ]
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("All split fractions must be positive.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")


def prepare_split_frame(
    frame: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    """Validate and chronologically sort split metadata without shuffling."""
    required = [
        config.address_column,
        config.ordering_column,
        config.label_column,
    ]
    require_columns(frame.columns, required)

    split_frame = frame.loc[:, required].copy()
    split_frame["row_index"] = frame.index.astype(int)
    split_frame[config.ordering_column] = pd.to_numeric(
        split_frame[config.ordering_column],
        errors="coerce",
    )

    if split_frame[config.ordering_column].isna().any():
        raise ValueError(f"{config.ordering_column} contains missing/non-numeric values.")
    if split_frame[config.address_column].isna().any():
        raise ValueError(f"{config.address_column} contains missing values.")
    if split_frame[config.label_column].isna().any():
        raise ValueError(f"{config.label_column} contains missing values.")

    sorted_frame = split_frame.sort_values(
        [config.ordering_column, "row_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    if not is_chronologically_ordered(sorted_frame[config.ordering_column]):
        raise ValueError("Sorted split frame is not chronologically ordered.")
    return sorted_frame


def assign_chronological_splits(
    sorted_frame: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    """Assign train/validation/test labels by chronological row position."""
    validate_split_config(config)
    total_rows = len(sorted_frame)
    if total_rows < 3:
        raise ValueError("At least three rows are required for a 3-way split.")

    train_end = int(total_rows * config.train_fraction)
    validation_end = train_end + int(total_rows * config.validation_fraction)
    if train_end <= 0 or validation_end <= train_end or validation_end >= total_rows:
        raise ValueError("Split fractions produced an empty partition.")

    result = sorted_frame.copy()
    result["split"] = "test"
    result.loc[: train_end - 1, "split"] = "train"
    result.loc[train_end : validation_end - 1, "split"] = "validation"
    assert_temporal_split_boundaries(result, config.ordering_column)
    return result


def assert_temporal_split_boundaries(
    split_frame: pd.DataFrame,
    ordering_column: str = "createdBlockNumber",
) -> None:
    """Assert chronological boundaries between train, validation, and test."""
    train = split_frame.loc[split_frame["split"] == "train", ordering_column]
    validation = split_frame.loc[split_frame["split"] == "validation", ordering_column]
    test = split_frame.loc[split_frame["split"] == "test", ordering_column]
    if train.empty or validation.empty or test.empty:
        raise AssertionError("All splits must be non-empty.")
    assert train.max() <= validation.min()
    assert validation.max() <= test.min()


def summarize_split_counts(
    split_frame: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> dict[str, dict[str, Any]]:
    """Return class counts and imbalance ratios for each split."""
    summaries: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation", "test"):
        subset = split_frame[split_frame["split"] == split_name]
        positives = int((subset[config.label_column] == config.positive_label).sum())
        negatives = int(len(subset) - positives)
        summaries[split_name] = {
            "rows": int(len(subset)),
            "positive_count": positives,
            "negative_count": negatives,
            "negative_to_positive_ratio": None
            if positives == 0
            else negatives / positives,
            "block_min": int(subset[config.ordering_column].min()),
            "block_max": int(subset[config.ordering_column].max()),
        }
    return summaries


def detect_cross_split_duplicates(
    frame: pd.DataFrame,
    split_membership: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    """Detect duplicate values that appear in more than one split."""
    require_columns(frame.columns, ["row_index", *columns])
    require_columns(split_membership.columns, ["row_index", "split"])

    merged = frame.loc[:, ["row_index", *columns]].merge(
        split_membership.loc[:, ["row_index", "split"]],
        on="row_index",
        how="inner",
        validate="one_to_one",
    )

    results: dict[str, Any] = {}
    for column in columns:
        series = merged.loc[:, ["row_index", "split", column]].dropna()
        if pd.api.types.is_string_dtype(series[column]) or series[column].dtype == object:
            series = series[series[column].astype(str).str.strip().ne("")]

        split_counts = series.groupby(column)["split"].nunique()
        crossing_values = split_counts[split_counts > 1].index
        crossing = series[series[column].isin(crossing_values)]
        rows_by_split = crossing.groupby("split").size().to_dict()
        results[column] = {
            "cross_split_duplicate_values": int(len(crossing_values)),
            "cross_split_duplicate_rows": int(len(crossing)),
            "rows_by_split": {
                split_name: int(rows_by_split.get(split_name, 0))
                for split_name in ("train", "validation", "test")
            },
        }
    return results


def build_temporal_split(
    config: SplitConfig = SplitConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build split membership and metadata for the configured raw dataset."""
    metadata_columns = [
        config.address_column,
        config.ordering_column,
        config.label_column,
    ]
    metadata = read_dataset_columns(metadata_columns, path=config.dataset)
    sorted_frame = prepare_split_frame(metadata, config)
    split_membership = assign_chronological_splits(sorted_frame, config)

    duplicate_columns = tuple(dict.fromkeys(config.duplicate_check_columns))
    duplicate_frame = read_dataset_columns(duplicate_columns, path=config.dataset)
    duplicate_frame["row_index"] = duplicate_frame.index.astype(int)

    metadata_json = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(config.dataset),
        "ordering_column": config.ordering_column,
        "split_fractions": {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
            "test": config.test_fraction,
        },
        "row_count": int(len(split_membership)),
        "split_counts": summarize_split_counts(split_membership, config),
        "temporal_boundary_assertions": {
            "max_train_block_lte_min_validation_block": True,
            "max_validation_block_lte_min_test_block": True,
        },
        "cross_split_duplicates": detect_cross_split_duplicates(
            duplicate_frame,
            split_membership,
            duplicate_columns,
        ),
    }
    return split_membership, metadata_json


def save_temporal_split(
    split_membership: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path = DEFAULT_SPLIT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Save compact split membership and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = output_dir / "temporal_split_membership.csv"
    metadata_path = output_dir / "temporal_split_metadata.json"

    columns = ["row_index", "address", "createdBlockNumber", "label", "split"]
    split_membership.loc[:, columns].to_csv(membership_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"membership": membership_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create chronological data splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG_PATH,
        help="YAML split configuration path.",
    )
    args = parser.parse_args()

    config = load_split_config(args.config)
    split_membership, metadata = build_temporal_split(config)
    paths = save_temporal_split(split_membership, metadata, config.output_dir)
    print(f"Wrote split membership to {paths['membership']}")
    print(f"Wrote split metadata to {paths['metadata']}")


if __name__ == "__main__":
    main()
