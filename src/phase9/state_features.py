"""Leakage-safe deployment-available creator state for Phase 9C."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


CREATOR_COLUMNS = [
    "prior_contract_count",
    "prior_verified_ponzi_count",
    "prior_verified_ponzi_flag",
    "creator_ponzi_rate",
    "blocks_since_creator_last_deployment",
    "prior_creator_positive_recency",
]


@dataclass(frozen=True)
class CreatorStateResult:
    train_raw: pd.DataFrame
    target_raw: pd.DataFrame
    train_model: np.ndarray
    target_model: np.ndarray


def _row(
    creator: str,
    block: int,
    counts: Counter[str],
    positive_counts: Counter[str],
    last_deployment: dict[str, int],
    last_positive: dict[str, int],
    sentinel: int,
) -> dict[str, float]:
    prior_count = int(counts[creator])
    prior_positive = int(positive_counts[creator])
    return {
        "prior_contract_count": prior_count,
        "prior_verified_ponzi_count": prior_positive,
        "prior_verified_ponzi_flag": int(prior_positive > 0),
        "creator_ponzi_rate": float(prior_positive / max(prior_count, 1)),
        "blocks_since_creator_last_deployment": int(block - last_deployment[creator])
        if creator in last_deployment
        else int(sentinel),
        "prior_creator_positive_recency": int(block - last_positive[creator])
        if creator in last_positive
        else int(sentinel),
    }


def transform_creator_features(raw: pd.DataFrame) -> np.ndarray:
    values = raw.loc[:, CREATOR_COLUMNS].astype(float).copy()
    for column in (
        "prior_contract_count",
        "prior_verified_ponzi_count",
        "blocks_since_creator_last_deployment",
        "prior_creator_positive_recency",
    ):
        values[column] = np.log1p(values[column])
    return values.to_numpy(dtype=np.float32)


def build_creator_state(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    sentinel: int,
) -> CreatorStateResult:
    """Build train-prequential and target-forward creator state.

    Training labels update verified state after each complete block. Target
    labels are never read. Label-free target deployment activity updates only
    after each complete target block.
    """
    required_train = {"row_index", "creator", "createdBlockNumber", "label"}
    required_target = {"row_index", "creator", "createdBlockNumber"}
    if not required_train.issubset(train.columns) or not required_target.issubset(target.columns):
        raise ValueError("Creator-state inputs are missing required columns.")

    counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    last_deployment: dict[str, int] = {}
    last_positive: dict[str, int] = {}
    train_rows: list[dict[str, float | int]] = []
    ordered_train = train.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
    for block, block_rows in ordered_train.groupby("createdBlockNumber", sort=True):
        block = int(block)
        for item in block_rows.itertuples(index=False):
            creator = str(item.creator)
            train_rows.append({"row_index": int(item.row_index), **_row(
                creator, block, counts, positive_counts, last_deployment,
                last_positive, sentinel,
            )})
        for item in block_rows.itertuples(index=False):
            creator = str(item.creator)
            counts[creator] += 1
            last_deployment[creator] = block
            if int(item.label) == 1:
                positive_counts[creator] += 1
                last_positive[creator] = block

    target_rows: list[dict[str, float | int]] = []
    ordered_target = target.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
    for block, block_rows in ordered_target.groupby("createdBlockNumber", sort=True):
        block = int(block)
        for item in block_rows.itertuples(index=False):
            creator = str(item.creator)
            target_rows.append({"row_index": int(item.row_index), **_row(
                creator, block, counts, positive_counts, last_deployment,
                last_positive, sentinel,
            )})
        for item in block_rows.itertuples(index=False):
            creator = str(item.creator)
            counts[creator] += 1
            last_deployment[creator] = block

    train_raw = pd.DataFrame(train_rows).set_index("row_index").reindex(train["row_index"]).reset_index()
    target_raw = pd.DataFrame(target_rows).set_index("row_index").reindex(target["row_index"]).reset_index()
    return CreatorStateResult(
        train_raw=train_raw,
        target_raw=target_raw,
        train_model=transform_creator_features(train_raw),
        target_model=transform_creator_features(target_raw),
    )
