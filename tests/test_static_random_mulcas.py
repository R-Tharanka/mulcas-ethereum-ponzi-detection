"""Focused integrity tests for the active random-split three-viewer workflow.

These tests exercise deterministic splitting, score aggregation, thresholding,
recorded evidence, and raw-data integrity. They deliberately avoid a full model
fit because the complete grid search is an experiment run, not a unit test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.static_random_mulcas import (
    DEFAULT_BASELINE,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT,
    best_f1_threshold,
    ensemble_probabilities,
    evaluate,
    split_data,
)


EXPECTED_DATASET_SHA256 = (
    "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"
)


# Build a small unique-address corpus that makes split membership auditable.
def _example_frame() -> pd.DataFrame:
    """Return a deterministic imbalanced binary corpus for split tests."""

    return pd.DataFrame(
        {
            "address": [f"0x{index:040x}" for index in range(100)],
            "opcode": [f"PUSH1 STOP TOKEN{index % 7}" for index in range(100)],
            "label": [1 if index < 20 else 0 for index in range(100)],
        }
    )


# Ensure the two-stage random split is stable and never duplicates a row.
def test_random_split_is_deterministic_and_disjoint() -> None:
    """Reproduce the 80/20 outer and 80/20 inner stratified split contract."""

    frame = _example_frame()
    first = split_data(frame, seed=42)
    second = split_data(frame, seed=42)
    train, fit, validation, test = first

    assert [len(part) for part in first] == [80, 64, 16, 20]
    assert [part.address.tolist() for part in first] == [
        part.address.tolist() for part in second
    ]
    assert set(fit.address).isdisjoint(validation.address)
    assert set(train.address).isdisjoint(test.address)
    assert set(fit.address) | set(validation.address) == set(train.address)


# Use tiny deterministic probability providers to verify equal-weight averaging.
def test_ensemble_probabilities_are_an_arithmetic_mean() -> None:
    """Give every viewer exactly one-third influence on the ensemble score."""

    class FixedModel:
        def __init__(self, positive_probabilities):
            self.positive_probabilities = np.asarray(positive_probabilities)

        def predict_proba(self, documents):
            assert len(documents) == len(self.positive_probabilities)
            return np.column_stack(
                [1.0 - self.positive_probabilities, self.positive_probabilities]
            )

    models = [FixedModel([0.2, 0.8]), FixedModel([0.4, 0.6]), FixedModel([0.6, 0.4])]
    scores = ensemble_probabilities(models, ["A", "B"])

    np.testing.assert_allclose(scores, [0.4, 0.6])


# Confirm that validation selection and fixed-threshold evaluation stay separate.
def test_threshold_selection_and_evaluation() -> None:
    """Select a perfect validation threshold and report the expected confusion matrix."""

    labels = np.asarray([0, 1, 1, 0])
    probabilities = np.asarray([0.1, 0.8, 0.7, 0.2])

    threshold, validation_f1 = best_f1_threshold(labels, probabilities)
    metrics = evaluate(labels, probabilities, threshold)

    assert threshold == pytest.approx(0.7)
    assert validation_f1 == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (
        2,
        0,
        0,
        2,
    )


# Anchor the active output to the collaborator's recorded random-split evidence.
def test_recorded_result_and_comparison_inputs_exist() -> None:
    """Validate the compact result schema and its key reported metrics."""

    assert DEFAULT_OUTPUT.is_file()
    assert DEFAULT_BASELINE.is_file()

    result = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert result["experiment"] == "random_stratified_static_opcode_three_viewer"
    assert result["seed"] == 42
    assert result["train_test_shared_addresses"] == 3
    assert result["test"]["precision"] == pytest.approx(0.8833333333333333)
    assert result["test"]["recall"] == pytest.approx(0.8412698412698413)
    assert result["test"]["f1"] == pytest.approx(0.8617886178861789)


# Hash the ignored raw CSV only when it is present in the local workspace.
def test_local_raw_dataset_hash_is_unchanged() -> None:
    """Protect the active 6,498-row source without requiring it in a fresh clone."""

    if not DEFAULT_DATASET.is_file():
        pytest.skip("Raw dataset is intentionally ignored by Git")

    digest = hashlib.sha256()
    with DEFAULT_DATASET.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    assert digest.hexdigest() == EXPECTED_DATASET_SHA256

