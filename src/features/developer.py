"""Chronological creator feature without future-label leakage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.validate import require_columns


@dataclass
class ChronologicalCreatorTransformer:
    """Flag creators known from strictly earlier positive training contracts."""

    creator_column: str = "creator"
    block_column: str = "createdBlockNumber"
    label_column: str = "label"
    positive_label: int = 1
    positive_creators_: set[str] = field(default_factory=set, init=False, repr=False)
    is_fitted_: bool = field(default=False, init=False)

    def _validated(self, frame: pd.DataFrame, require_label: bool) -> pd.DataFrame:
        required = [self.creator_column, self.block_column]
        if require_label:
            required.append(self.label_column)
        require_columns(frame.columns, required)
        if frame[required].isna().any().any():
            raise ValueError("Creator feature columns must not contain null values.")
        result = frame.copy()
        result[self.block_column] = pd.to_numeric(result[self.block_column], errors="raise")
        return result

    def fit(self, frame: pd.DataFrame) -> "ChronologicalCreatorTransformer":
        """Learn positive creators from the training partition only."""
        data = self._validated(frame, require_label=True)
        self.positive_creators_ = set(
            data.loc[data[self.label_column] == self.positive_label, self.creator_column]
            .astype(str)
            .tolist()
        )
        self.is_fitted_ = True
        return self

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Create prequential training features and then retain final train state.

        Labels from a block are added only after every contract in that block has
        been represented, so equal-block contracts cannot leak labels to peers.
        """
        data = self._validated(frame, require_label=True)
        ordered = data.assign(_position=np.arange(len(data))).sort_values(
            [self.block_column, "_position"], kind="mergesort"
        )
        features = np.zeros((len(data), 1), dtype=np.float32)
        known_positive_creators: set[str] = set()

        for _, block_rows in ordered.groupby(self.block_column, sort=True):
            positions = block_rows["_position"].to_numpy(dtype=int)
            creators = block_rows[self.creator_column].astype(str)
            features[positions, 0] = creators.isin(known_positive_creators).astype(np.float32)
            new_positive_creators = block_rows.loc[
                block_rows[self.label_column] == self.positive_label,
                self.creator_column,
            ].astype(str)
            known_positive_creators.update(new_positive_creators)

        self.positive_creators_ = known_positive_creators
        self.is_fitted_ = True
        return features

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform later data from frozen training state without reading labels."""
        if not self.is_fitted_:
            raise RuntimeError("ChronologicalCreatorTransformer must be fitted first.")
        data = self._validated(frame, require_label=False)
        return (
            data[self.creator_column]
            .astype(str)
            .isin(self.positive_creators_)
            .to_numpy(dtype=np.float32)
            .reshape(-1, 1)
        )
