"""Chronological period construction for temporal robustness evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalScenario:
    """One expanding-history train period and its immediately later target."""

    scenario_id: str
    train_periods: tuple[str, ...]
    prediction_period: str


def build_ponzi_occurrence_periods(
    frame: pd.DataFrame,
    *,
    positive_ends: Sequence[int] = (50, 100, 150, 200, 250),
    expected_positives: int = 314,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Assign all rows to whole-block periods delimited by positive occurrences."""
    required = {"row_index", "createdBlockNumber", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing period columns: {sorted(missing)}")
    if frame[sorted(required)].isna().any().any():
        raise ValueError("Period columns must not contain null values.")

    ordered = frame.copy()
    ordered["createdBlockNumber"] = pd.to_numeric(
        ordered["createdBlockNumber"], errors="raise"
    )
    ordered["label"] = pd.to_numeric(ordered["label"], errors="raise").astype(int)
    ordered = ordered.sort_values(
        ["createdBlockNumber", "row_index"], kind="mergesort"
    ).reset_index(drop=True)
    if not ordered["row_index"].is_unique:
        raise ValueError("row_index must be unique.")
    if not set(ordered["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1 values.")

    positives = ordered.loc[ordered["label"] == 1]
    if len(positives) != expected_positives:
        raise ValueError(
            f"Expected {expected_positives} positive contracts, found {len(positives)}."
        )
    ends = tuple(int(value) for value in positive_ends)
    if not ends or any(value <= 0 for value in ends):
        raise ValueError("Positive occurrence boundaries must be positive.")
    if list(ends) != sorted(set(ends)) or ends[-1] >= expected_positives:
        raise ValueError("Positive occurrence boundaries must be unique and increasing.")

    boundary_blocks = [
        int(positives.iloc[position - 1]["createdBlockNumber"])
        for position in ends
    ]
    if boundary_blocks != sorted(boundary_blocks):
        raise AssertionError("Positive boundary blocks are not chronological.")

    block_values = ordered["createdBlockNumber"].to_numpy()
    period_numbers = np.searchsorted(boundary_blocks, block_values, side="left")
    ordered["period"] = [f"P{number}" for number in period_numbers]

    definitions: list[dict[str, Any]] = []
    expected_counts = np.diff((0, *ends, expected_positives)).tolist()
    for period_number, expected_count in enumerate(expected_counts):
        name = f"P{period_number}"
        rows = ordered.loc[ordered["period"] == name]
        if rows.empty:
            raise AssertionError(f"Temporal period {name} is empty.")
        positive_count = int(rows["label"].sum())
        negative_count = int(len(rows) - positive_count)
        definitions.append(
            {
                "period": name,
                "row_count": int(len(rows)),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "negative_to_positive_ratio": (
                    float(negative_count / positive_count)
                    if positive_count
                    else None
                ),
                "block_min": int(rows["createdBlockNumber"].min()),
                "block_max": int(rows["createdBlockNumber"].max()),
                "intended_positive_count": int(expected_count),
                "positive_count_difference": int(positive_count - expected_count),
                "boundary_block": (
                    boundary_blocks[period_number]
                    if period_number < len(boundary_blocks)
                    else None
                ),
            }
        )

    if ordered["period"].isna().any() or len(ordered) != len(frame):
        raise AssertionError("Period assignment did not cover the complete dataset.")
    if ordered.groupby("createdBlockNumber")["period"].nunique().max() != 1:
        raise AssertionError("A creation block was split across periods.")
    for previous, current in zip(definitions, definitions[1:], strict=False):
        if previous["block_max"] >= current["block_min"]:
            raise AssertionError("Temporal periods are not strictly chronological.")
    return ordered, definitions


def incremental_scenarios() -> tuple[TemporalScenario, ...]:
    """Return the four predeclared expanding-history Phase 7A scenarios."""
    return tuple(
        TemporalScenario(
            scenario_id=f"train_P0_to_P{target - 1}__predict_P{target}",
            train_periods=tuple(f"P{value}" for value in range(target)),
            prediction_period=f"P{target}",
        )
        for target in range(2, 6)
    )


def validate_scenario_chronology(
    assigned: pd.DataFrame, scenario: TemporalScenario
) -> None:
    """Assert that every training block strictly precedes the target period."""
    train = assigned.loc[assigned["period"].isin(scenario.train_periods)]
    target = assigned.loc[assigned["period"] == scenario.prediction_period]
    if train.empty or target.empty:
        raise ValueError(f"Scenario {scenario.scenario_id} has an empty partition.")
    if int(train["createdBlockNumber"].max()) >= int(
        target["createdBlockNumber"].min()
    ):
        raise AssertionError(f"Scenario {scenario.scenario_id} is not forward-only.")
