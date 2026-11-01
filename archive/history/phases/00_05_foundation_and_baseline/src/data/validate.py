"""Reusable validation primitives for dataset integrity audits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd


def missing_columns(columns: Iterable[str], required_columns: Sequence[str]) -> list[str]:
    """Return required columns not present in ``columns``."""
    present = set(columns)
    return [column for column in required_columns if column not in present]


def require_columns(columns: Iterable[str], required_columns: Sequence[str]) -> None:
    """Raise ``ValueError`` if required columns are missing."""
    missing = missing_columns(columns, required_columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def unexpected_values(values: pd.Series, allowed_values: set[Any]) -> list[Any]:
    """Return non-null values outside an allowed set."""
    observed = set(values.dropna().unique().tolist())
    return sorted(observed - allowed_values)


def missing_or_empty_count(values: pd.Series) -> int:
    """Count nulls and strings that become empty after stripping whitespace."""
    missing = values.isna()
    if pd.api.types.is_string_dtype(values) or values.dtype == object:
        empty = values.fillna("").astype(str).str.strip().eq("")
        return int((missing | empty).sum())
    return int(missing.sum())


def duplicate_value_count(values: pd.Series, ignore_missing: bool = True) -> int:
    """Count rows whose value duplicates another row."""
    series = values
    if ignore_missing:
        series = series.dropna()
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            series = series[series.astype(str).str.strip().ne("")]
    return int(series.duplicated(keep=False).sum())


def duplicate_group_count(
    frame: pd.DataFrame,
    subset: Sequence[str],
    ignore_missing: bool = True,
) -> int:
    """Count duplicated rows by a subset of columns."""
    data = frame.loc[:, list(subset)]
    if ignore_missing:
        data = data.dropna()
        for column in data.columns:
            if pd.api.types.is_string_dtype(data[column]) or data[column].dtype == object:
                data = data[data[column].astype(str).str.strip().ne("")]
    return int(data.duplicated(subset=list(subset), keep=False).sum())


def is_chronologically_ordered(values: pd.Series) -> bool:
    """Return whether non-null values are monotonically non-decreasing."""
    return bool(values.dropna().is_monotonic_increasing)
