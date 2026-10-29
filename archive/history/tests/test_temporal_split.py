"""Tests for leakage-safe chronological splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import (
    SplitConfig,
    assert_temporal_split_boundaries,
    assign_chronological_splits,
    detect_cross_split_duplicates,
    prepare_split_frame,
    summarize_split_counts,
)


def _config() -> SplitConfig:
    return SplitConfig(
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        duplicate_check_columns=("address", "opcode", "contractCode"),
    )


def test_split_sorts_by_block_without_shuffling() -> None:
    frame = pd.DataFrame(
        {
            "address": ["late", "early", "middle", "latest"],
            "createdBlockNumber": [40, 10, 20, 50],
            "label": [0, 1, 0, 1],
        }
    )

    split_frame = assign_chronological_splits(prepare_split_frame(frame, _config()), _config())

    assert split_frame["address"].tolist() == ["early", "middle", "late", "latest"]
    assert split_frame["split"].tolist() == ["train", "train", "validation", "test"]
    assert_temporal_split_boundaries(split_frame)


def test_boundary_assertion_catches_future_data_leakage() -> None:
    leaked = pd.DataFrame(
        {
            "createdBlockNumber": [30, 10, 20],
            "split": ["train", "validation", "test"],
        }
    )

    with pytest.raises(AssertionError):
        assert_temporal_split_boundaries(leaked)


def test_split_summary_reports_class_imbalance_ratio() -> None:
    split_frame = pd.DataFrame(
        {
            "address": ["a", "b", "c", "d"],
            "createdBlockNumber": [1, 2, 3, 4],
            "label": [0, 1, 0, 0],
            "split": ["train", "train", "validation", "test"],
        }
    )

    summary = summarize_split_counts(split_frame, _config())

    assert summary["train"]["positive_count"] == 1
    assert summary["train"]["negative_count"] == 1
    assert summary["train"]["negative_to_positive_ratio"] == 1.0
    assert summary["validation"]["negative_to_positive_ratio"] is None


def test_cross_split_duplicate_detection_flags_shared_values() -> None:
    frame = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "address": ["addr-a", "addr-b", "addr-a", "addr-c"],
            "opcode": ["op-1", "op-2", "op-3", "op-2"],
            "contractCode": ["byte-a", "byte-b", "byte-c", "byte-b"],
        }
    )
    split_membership = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "split": ["train", "train", "validation", "test"],
        }
    )

    duplicates = detect_cross_split_duplicates(
        frame,
        split_membership,
        ("address", "opcode", "contractCode"),
    )

    assert duplicates["address"]["cross_split_duplicate_values"] == 1
    assert duplicates["opcode"]["cross_split_duplicate_values"] == 1
    assert duplicates["contractCode"]["cross_split_duplicate_values"] == 1
