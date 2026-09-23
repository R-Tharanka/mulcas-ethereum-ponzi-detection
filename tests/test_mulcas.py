"""Unit tests for MulCas cascade routing and final voting."""

from __future__ import annotations

import pandas as pd

from src.models.mulcas import MulCasCascade, ViewerPrediction, ViewerSpec


def _prediction(index: list[int], probabilities: list[float]) -> ViewerPrediction:
    probability = pd.Series(probabilities, index=index)
    return ViewerPrediction(
        probability=probability,
        prediction=(probability >= 0.5).astype(int),
    )


def test_cascade_routes_only_unresolved_samples_and_preserves_order() -> None:
    indices = [42, 7, 99]
    cascade = MulCasCascade(
        [
            ViewerSpec("first", 0.05, 0.95),
            ViewerSpec("second", 0.05, 0.51),
        ],
        voting_viewers=["first", "second"],
        voting_threshold=0,
    )
    result = cascade.predict(
        indices,
        cascade_predictions={
            "first": _prediction(indices, [0.95, 0.4, 0.5]),
            "second": _prediction(indices, [0.1, 0.01, 0.85]),
        },
        voting_predictions={
            "first": _prediction(indices, [0.95, 0.4, 0.5]),
            "second": _prediction(indices, [0.1, 0.01, 0.85]),
        },
    )

    assert result.predictions["sample_index"].tolist() == indices
    assert result.predictions["handled_by"].tolist() == ["vote", "second", "second"]
    second_stage = result.traces[
        (result.traces["phase"] == "cascade")
        & (result.traces["viewer"] == "second")
    ]
    assert second_stage[
        "sample_index"
    ].tolist() == [42, 7, 99]


def test_unresolved_samples_use_explicit_voting_threshold() -> None:
    indices = [10, 20]
    cascade = MulCasCascade(
        [
            ViewerSpec("one", 0.01, 0.99),
        ],
        voting_viewers=["one", "two", "three"],
        voting_threshold=1,
    )
    result = cascade.predict(
        indices,
        cascade_predictions={
            "one": _prediction(indices, [0.6, 0.6]),
        },
        voting_predictions={
            "one": _prediction(indices, [0.6, 0.6]),
            "two": _prediction(indices, [0.7, 0.4]),
            "three": _prediction(indices, [0.4, 0.3]),
        },
    )

    assert result.predictions["prediction"].tolist() == [1, 0]
    assert result.predictions["handled_by"].tolist() == ["vote", "vote"]
    vote_trace = result.traces[result.traces["viewer"] == "vote"]
    assert vote_trace["positive_votes"].tolist() == [2, 1]


def test_invalid_or_missing_viewer_data_is_rejected() -> None:
    cascade = MulCasCascade(
        [ViewerSpec("only", 0.05, 0.95)],
        voting_viewers=["only"],
        voting_threshold=0,
    )

    try:
        cascade.predict([1], cascade_predictions={}, voting_predictions={})
    except ValueError as error:
        assert "Missing predictions" in str(error)
    else:
        raise AssertionError("Missing viewer predictions should be rejected.")
