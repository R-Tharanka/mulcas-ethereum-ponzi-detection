"""Dataset loading helpers for the released MulCas/Ponzi CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.paths import RAW_DATA_DIR


DEFAULT_RAW_DATASET_PATH: Path = RAW_DATA_DIR / "PonziDataset_20221114.csv"

LARGE_TEXT_COLUMNS: tuple[str, ...] = (
    "opcode",
    "code",
    "contractCode",
    "creationCode",
)


def read_dataset_header(path: Path = DEFAULT_RAW_DATASET_PATH) -> list[str]:
    """Return CSV column names without loading data rows."""
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_dataset_columns(
    columns: Iterable[str],
    path: Path = DEFAULT_RAW_DATASET_PATH,
) -> pd.DataFrame:
    """Read selected columns from the raw dataset."""
    return pd.read_csv(path, usecols=list(columns), low_memory=False)


def read_dataset_metadata(path: Path = DEFAULT_RAW_DATASET_PATH) -> pd.DataFrame:
    """Read non-text columns for integrity checks that do not need code fields."""
    columns = [
        column
        for column in read_dataset_header(path)
        if column not in LARGE_TEXT_COLUMNS
    ]
    return read_dataset_columns(columns, path=path)
