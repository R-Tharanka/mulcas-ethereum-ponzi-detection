"""Dataset integrity audit for the released raw Ponzi dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load import (
    DEFAULT_RAW_DATASET_PATH,
    LARGE_TEXT_COLUMNS,
    read_dataset_columns,
    read_dataset_header,
    read_dataset_metadata,
)
from src.data.validate import (
    duplicate_group_count,
    duplicate_value_count,
    is_chronologically_ordered,
    missing_or_empty_count,
    require_columns,
    unexpected_values,
)
from src.utils.paths import RESULTS_DIR


PUBLISHED_TOTAL_CONTRACTS = 6_498
PUBLISHED_PONZI_CONTRACTS = 314

EXPECTED_COLUMNS: tuple[str, ...] = (
    "address",
    "opcode",
    "code",
    "contractCode",
    "createdBlockNumber",
    "createdTransactionHash",
    "creationCode",
    "creator",
    "label",
    "timestamp",
)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 without modifying the input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _stringify_keys(values: dict[Any, Any]) -> dict[str, Any]:
    return {str(_json_ready(key)): _json_ready(value) for key, value in values.items()}


def _range_summary(values: pd.Series) -> dict[str, Any]:
    non_null = values.dropna()
    if non_null.empty:
        return {"min": None, "max": None}
    minimum = _json_ready(non_null.min())
    maximum = _json_ready(non_null.max())
    if isinstance(minimum, float) and minimum.is_integer():
        minimum = int(minimum)
    if isinstance(maximum, float) and maximum.is_integer():
        maximum = int(maximum)
    return {
        "min": minimum,
        "max": maximum,
    }


def _top_counts(values: pd.Series, limit: int = 10) -> list[dict[str, Any]]:
    counts = values.value_counts(dropna=True).head(limit)
    return [
        {"value": _json_ready(index), "count": int(count)}
        for index, count in counts.items()
    ]


def _opcode_audit(path: Path) -> dict[str, Any]:
    opcode = read_dataset_columns(["opcode"], path=path)["opcode"]
    missing_or_empty = missing_or_empty_count(opcode)
    non_empty = opcode.dropna()
    non_empty = non_empty[non_empty.astype(str).str.strip().ne("")]
    duplicate_rows = duplicate_value_count(non_empty, ignore_missing=True)
    duplicate_groups = int((non_empty.value_counts(dropna=True) > 1).sum())
    return {
        "missing_or_empty_count": missing_or_empty,
        "exact_duplicate_rows": duplicate_rows,
        "exact_duplicate_groups": duplicate_groups,
        "memory_usage_bytes_deep": int(opcode.memory_usage(deep=True)),
    }


def _null_counts(path: Path, columns: list[str], chunksize: int = 500) -> dict[str, int]:
    counts = {column: 0 for column in columns}
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        for column, count in chunk.isna().sum().items():
            counts[column] += int(count)
    return counts


def build_dataset_audit(path: Path = DEFAULT_RAW_DATASET_PATH) -> dict[str, Any]:
    """Build the Phase 1 integrity audit as a machine-readable dictionary."""
    path = Path(path)
    columns = read_dataset_header(path)
    require_columns(columns, EXPECTED_COLUMNS)

    metadata = read_dataset_metadata(path)
    metadata["createdBlockNumber"] = pd.to_numeric(
        metadata["createdBlockNumber"],
        errors="coerce",
    )
    timestamps = pd.to_datetime(metadata["timestamp"], errors="coerce")

    label_counts = metadata["label"].value_counts(dropna=False).sort_index()
    positive_count = int((metadata["label"] == 1).sum())
    creators_with_multiple_contracts = metadata["creator"].value_counts(dropna=True)
    positive_creators = metadata.loc[metadata["label"] == 1, "creator"].value_counts(
        dropna=True
    )

    text_sample = pd.read_csv(path, usecols=list(LARGE_TEXT_COLUMNS), nrows=25)
    text_dtypes = {
        column: str(dtype)
        for column, dtype in text_sample.dtypes.items()
    }
    opcode_audit = _opcode_audit(path)

    audit = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": compute_sha256(path),
        "published_reference": {
            "total_contracts": PUBLISHED_TOTAL_CONTRACTS,
            "ponzi_contracts": PUBLISHED_PONZI_CONTRACTS,
        },
        "row_count": int(len(metadata)),
        "columns": columns,
        "dtypes": {
            **{column: str(dtype) for column, dtype in metadata.dtypes.items()},
            **text_dtypes,
        },
        "label_values": sorted(
            [_json_ready(value) for value in metadata["label"].dropna().unique()]
        ),
        "unexpected_label_values": unexpected_values(metadata["label"], {0, 1}),
        "class_counts": _stringify_keys(label_counts.to_dict()),
        "null_counts": _null_counts(path, columns),
        "duplicate_addresses": duplicate_value_count(metadata["address"]),
        "duplicate_transaction_hashes": duplicate_value_count(
            metadata["createdTransactionHash"]
        ),
        "duplicate_block_address_combinations": duplicate_group_count(
            metadata,
            ["createdBlockNumber", "address"],
        ),
        "block_number_range": _range_summary(metadata["createdBlockNumber"]),
        "timestamp_range": {
            "min": None if timestamps.dropna().empty else timestamps.min().isoformat(),
            "max": None if timestamps.dropna().empty else timestamps.max().isoformat(),
        },
        "chronological_ordering": {
            "by_createdBlockNumber": is_chronologically_ordered(
                metadata["createdBlockNumber"]
            ),
            "by_timestamp": is_chronologically_ordered(timestamps),
        },
        "unique_creators": int(metadata["creator"].nunique(dropna=True)),
        "creators_with_multiple_contracts": int(
            (creators_with_multiple_contracts > 1).sum()
        ),
        "creators_with_multiple_positive_contracts": int((positive_creators > 1).sum()),
        "top_creators_by_contract_count": _top_counts(metadata["creator"]),
        "top_creators_by_positive_contract_count": _top_counts(
            metadata.loc[metadata["label"] == 1, "creator"]
        ),
        "opcode": opcode_audit,
        "memory_usage_bytes": {
            "metadata_only_deep": int(metadata.memory_usage(deep=True).sum()),
            "opcode_deep": opcode_audit["memory_usage_bytes_deep"],
            "raw_csv_file_size": path.stat().st_size,
        },
        "paper_discrepancies": {
            "row_count_delta": int(len(metadata) - PUBLISHED_TOTAL_CONTRACTS),
            "ponzi_count_delta": int(positive_count - PUBLISHED_PONZI_CONTRACTS),
        },
    }
    audit["memory_usage_bytes"]["metadata_plus_opcode_deep"] = (
        audit["memory_usage_bytes"]["metadata_only_deep"]
        + audit["memory_usage_bytes"]["opcode_deep"]
    )
    return audit


def save_dataset_audit(
    audit: dict[str, Any],
    output_dir: Path = RESULTS_DIR / "baseline",
) -> Path:
    """Save the audit JSON under the baseline results directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_audit.json"
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the raw Ponzi dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_RAW_DATASET_PATH,
        help="Path to the raw Ponzi CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "baseline",
        help="Directory for machine-readable audit output.",
    )
    args = parser.parse_args()

    output_path = save_dataset_audit(
        build_dataset_audit(args.dataset),
        output_dir=args.output_dir,
    )
    print(f"Wrote dataset audit to {output_path}")


if __name__ == "__main__":
    main()
