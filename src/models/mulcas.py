"""Reusable confidence cascade and final voting for MulCas reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ViewerSpec:
    """One configured viewer in cascade order."""

    name: str
    negative_threshold: float
    positive_threshold: float


@dataclass(frozen=True)
class ViewerPrediction:
    """Index-aligned predictions and positive-class probabilities."""

    probability: pd.Series
    prediction: pd.Series


@dataclass(frozen=True)
class CascadeResult:
    """Final predictions plus sample-level and viewer-level traces."""

    predictions: pd.DataFrame
    traces: pd.DataFrame


class MulCasCascade:
    """Route confident samples through N viewers, then vote on residuals."""

    def __init__(
        self,
        cascade_viewers: Sequence[ViewerSpec],
        voting_viewers: Sequence[str],
        voting_threshold: float,
    ) -> None:
        if not cascade_viewers:
            raise ValueError("At least one cascade viewer is required.")
        if not voting_viewers:
            raise ValueError("At least one voting viewer is required.")
        if voting_threshold < 0:
            raise ValueError("voting_threshold must be non-negative.")
        for viewer in cascade_viewers:
            valid = (
                0.0
                <= viewer.negative_threshold
                < viewer.positive_threshold
                <= 1.0
            )
            if not valid:
                raise ValueError(
                    "Viewer thresholds must satisfy 0 <= negative < positive <= 1."
                )
        self.cascade_viewers = tuple(cascade_viewers)
        self.voting_viewers = tuple(voting_viewers)
        self.voting_threshold = voting_threshold

    @staticmethod
    def _align(
        names: Sequence[str],
        predictions: Mapping[str, ViewerPrediction],
        ordered_index: pd.Index,
    ) -> dict[str, ViewerPrediction]:
        aligned: dict[str, ViewerPrediction] = {}
        for name in names:
            if name not in predictions:
                raise ValueError(f"Missing predictions for viewer: {name}")
            output = predictions[name]
            probability = output.probability.reindex(ordered_index)
            prediction = output.prediction.reindex(ordered_index)
            if probability.isna().any() or prediction.isna().any():
                raise ValueError(f"Viewer {name} is missing sample indices.")
            if not probability.between(0.0, 1.0).all():
                raise ValueError(f"Viewer {name} has invalid probabilities.")
            aligned[name] = ViewerPrediction(
                probability=probability.astype(float),
                prediction=prediction.astype(int),
            )
        return aligned

    def predict(
        self,
        sample_indices: Sequence[object],
        cascade_predictions: Mapping[str, ViewerPrediction],
        voting_predictions: Mapping[str, ViewerPrediction],
    ) -> CascadeResult:
        """Predict while preserving the exact supplied sample index order."""
        ordered_index = pd.Index(sample_indices, name="sample_index")
        if not ordered_index.is_unique:
            raise ValueError("sample_indices must be unique.")
        cascade_outputs = self._align(
            [viewer.name for viewer in self.cascade_viewers],
            cascade_predictions,
            ordered_index,
        )
        voting_outputs = self._align(
            self.voting_viewers,
            voting_predictions,
            ordered_index,
        )

        unresolved = pd.Series(True, index=ordered_index)
        final_prediction = pd.Series(-1, index=ordered_index, dtype=int)
        final_probability = pd.Series(np.nan, index=ordered_index, dtype=float)
        handled_by = pd.Series("", index=ordered_index, dtype=object)
        trace_rows: list[dict[str, object]] = []

        for stage, viewer in enumerate(self.cascade_viewers, start=1):
            output = cascade_outputs[viewer.name]
            active_indices = ordered_index[unresolved.to_numpy()]
            for sample_index in active_indices:
                probability = float(output.probability.loc[sample_index])
                prediction = int(output.prediction.loc[sample_index])
                accepted = (
                    probability < viewer.negative_threshold
                    or probability > viewer.positive_threshold
                )
                trace_rows.append(
                    {
                        "sample_index": sample_index,
                        "phase": "cascade",
                        "stage": stage,
                        "viewer": viewer.name,
                        "probability": probability,
                        "prediction": prediction,
                        "negative_threshold": viewer.negative_threshold,
                        "positive_threshold": viewer.positive_threshold,
                        "accepted": accepted,
                    }
                )
                if accepted:
                    final_prediction.loc[sample_index] = prediction
                    final_probability.loc[sample_index] = probability
                    handled_by.loc[sample_index] = viewer.name
                    unresolved.loc[sample_index] = False

        unresolved_indices = ordered_index[unresolved.to_numpy()]
        for sample_index in unresolved_indices:
            votes = np.array(
                [
                    voting_outputs[name].prediction.loc[sample_index]
                    for name in self.voting_viewers
                ],
                dtype=int,
            )
            probabilities = np.array(
                [
                    voting_outputs[name].probability.loc[sample_index]
                    for name in self.voting_viewers
                ],
                dtype=float,
            )
            positive_votes = int(votes.sum())
            for name, probability, prediction in zip(
                self.voting_viewers,
                probabilities,
                votes,
                strict=True,
            ):
                trace_rows.append(
                    {
                        "sample_index": sample_index,
                        "phase": "voting_viewer",
                        "stage": len(self.cascade_viewers) + 1,
                        "viewer": name,
                        "probability": float(probability),
                        "prediction": int(prediction),
                        "negative_threshold": np.nan,
                        "positive_threshold": np.nan,
                        "accepted": False,
                    }
                )
            final_prediction.loc[sample_index] = int(
                positive_votes > self.voting_threshold
            )
            final_probability.loc[sample_index] = float(probabilities.max())
            handled_by.loc[sample_index] = "vote"
            trace_rows.append(
                {
                    "sample_index": sample_index,
                    "phase": "vote_result",
                    "stage": len(self.cascade_viewers) + 1,
                    "viewer": "vote",
                    "probability": float(probabilities.max()),
                    "prediction": int(final_prediction.loc[sample_index]),
                    "negative_threshold": np.nan,
                    "positive_threshold": np.nan,
                    "accepted": True,
                    "positive_votes": positive_votes,
                    "total_votes": int(len(votes)),
                    "voting_threshold": self.voting_threshold,
                }
            )

        predictions = pd.DataFrame(
            {
                "sample_index": ordered_index,
                "prediction": final_prediction.to_numpy(),
                "probability": final_probability.to_numpy(),
                "handled_by": handled_by.to_numpy(),
            }
        )
        return CascadeResult(predictions=predictions, traces=pd.DataFrame(trace_rows))
