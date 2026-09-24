"""Forward-only temporal cross-validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_end_block: int
    validation_end_block: int


def make_positive_occurrence_folds(
    training_frame: pd.DataFrame,
    train_positive_ends: Sequence[int],
    validation_positive_ends: Sequence[int],
    *,
    block_column: str = "createdBlockNumber",
    label_column: str = "label",
    positive_label: int = 1,
) -> list[TemporalFold]:
    """Create expanding-window folds with whole-block positive boundaries."""
    if len(train_positive_ends) != len(validation_positive_ends):
        raise ValueError("Train and validation boundary lists must have equal length.")
    required = {"row_index", block_column, label_column}
    if not required.issubset(training_frame.columns):
        raise ValueError(f"Training frame must contain {sorted(required)}.")
    ordered = training_frame.sort_values(
        [block_column, "row_index"], kind="mergesort"
    ).reset_index(drop=True)
    positives = ordered[ordered[label_column] == positive_label]
    folds: list[TemporalFold] = []
    for fold_number, (train_end, validation_end) in enumerate(
        zip(train_positive_ends, validation_positive_ends, strict=True), start=1
    ):
        if not 0 < train_end < validation_end <= len(positives):
            raise ValueError("Each fold must have increasing positive boundaries.")
        train_block = int(positives.iloc[train_end - 1][block_column])
        validation_block = int(positives.iloc[validation_end - 1][block_column])
        train = ordered[ordered[block_column] <= train_block]
        validation = ordered[
            (ordered[block_column] > train_block)
            & (ordered[block_column] <= validation_block)
        ]
        if train.empty or validation.empty:
            raise ValueError("Temporal fold produced an empty partition.")
        if train[block_column].max() >= validation[block_column].min():
            raise AssertionError("Temporal CV training must precede validation.")
        if set(train["row_index"]) & set(validation["row_index"]):
            raise AssertionError("Temporal CV partitions must be disjoint.")
        folds.append(
            TemporalFold(
                fold=fold_number,
                train_indices=tuple(train["row_index"].astype(int)),
                validation_indices=tuple(validation["row_index"].astype(int)),
                train_end_block=train_block,
                validation_end_block=validation_block,
            )
        )
    return folds
