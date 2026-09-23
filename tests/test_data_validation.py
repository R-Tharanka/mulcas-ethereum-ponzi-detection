"""Unit tests for dataset validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate import (
    duplicate_group_count,
    duplicate_value_count,
    is_chronologically_ordered,
    missing_columns,
    missing_or_empty_count,
    require_columns,
    unexpected_values,
)


def test_missing_columns_reports_required_columns_in_order() -> None:
    assert missing_columns(["address", "label"], ["address", "creator", "label"]) == [
        "creator"
    ]


def test_require_columns_raises_for_missing_column() -> None:
    with pytest.raises(ValueError, match="creator"):
        require_columns(["address", "label"], ["address", "creator", "label"])


def test_unexpected_values_ignores_nulls() -> None:
    values = pd.Series([0, 1, 1, None, 2])

    assert unexpected_values(values, {0, 1}) == [2.0]


def test_missing_or_empty_count_handles_blank_strings() -> None:
    values = pd.Series(["PUSH1", "", "   ", None])

    assert missing_or_empty_count(values) == 3


def test_duplicate_value_count_ignores_missing_and_empty_values() -> None:
    values = pd.Series(["a", "a", "", None, "b"])

    assert duplicate_value_count(values) == 2


def test_duplicate_group_count_counts_rows_in_duplicate_groups() -> None:
    frame = pd.DataFrame(
        {
            "createdBlockNumber": [1, 1, 2],
            "address": ["0xabc", "0xabc", "0xdef"],
        }
    )

    assert duplicate_group_count(frame, ["createdBlockNumber", "address"]) == 2


def test_is_chronologically_ordered_allows_equal_adjacent_values() -> None:
    assert is_chronologically_ordered(pd.Series([1, 2, 2, 3]))
    assert not is_chronologically_ordered(pd.Series([1, 3, 2]))
