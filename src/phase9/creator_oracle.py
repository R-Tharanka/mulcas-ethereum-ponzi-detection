"""Non-deployable true-label online creator-state oracle for Phase 9A."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.evaluation.metrics import binary_classification_metrics
from src.evaluation.temporal import TemporalScenario
from src.experiments.phase7a import (
    build_historical_feature_matrices,
    fit_predict_frozen_mulcas,
)


def online_true_label_creator_flags(
    train: pd.DataFrame, target: pd.DataFrame
) -> np.ndarray:
    """Flag known-positive creators, revealing target labels only after each block."""
    positive_creators = set(train.loc[train["label"] == 1, "creator"].astype(str))
    ordered = target.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
    flags = pd.Series(0.0, index=target.index)
    for _, block in ordered.groupby("createdBlockNumber", sort=True):
        flags.loc[block.index] = block["creator"].astype(str).isin(positive_creators).astype(float)
        positive_creators.update(block.loc[block["label"] == 1, "creator"].astype(str))
    return flags.loc[target.index].to_numpy(dtype=float)


def replace_creator_column(matrix: csr_matrix, values: np.ndarray) -> csr_matrix:
    """Replace the final frozen creator column without touching static features."""
    if matrix.shape[0] != len(values):
        raise ValueError("Creator values must align one-to-one with matrix rows.")
    return hstack([matrix[:, :-1], csr_matrix(values.reshape(-1, 1))], format="csr")


def _prediction_frame(
    predictions: pd.DataFrame,
    target: pd.DataFrame,
    condition: str,
    frozen_flags: np.ndarray,
    oracle_flags: np.ndarray,
) -> pd.DataFrame:
    return predictions.rename(
        columns={
            "sample_index": "row_index",
            "prediction": f"{condition}_prediction",
            "probability": f"{condition}_score",
            "handled_by": f"{condition}_accepted_viewer",
        }
    ).merge(
        target[["row_index", "address", "creator", "createdBlockNumber", "label", "period"]],
        on="row_index",
        validate="one_to_one",
    ).assign(frozen_creator_flag=frozen_flags, oracle_creator_flag=oracle_flags)


def run_creator_oracle(
    assigned: pd.DataFrame,
    scenarios: tuple[TemporalScenario, ...],
    feature_config: dict[str, Any],
    model_config: dict[str, Any],
    mulcas_config: dict[str, Any],
    archived_predictions: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replay frozen MulCas with only the target creator state changed by the oracle."""
    metric_rows: list[dict[str, Any]] = []
    changed_rows: list[pd.DataFrame] = []
    replay_checks: dict[str, bool] = {}
    views = list(dict.fromkeys(v["feature_view"] for v in mulcas_config["viewer_sequence"]))

    for scenario in scenarios:
        train = assigned.loc[assigned["period"].isin(scenario.train_periods)].copy()
        target = assigned.loc[assigned["period"] == scenario.prediction_period].copy()
        train = train.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
        target = target.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
        matrices, _, frozen_flags = build_historical_feature_matrices(
            train, target, views, ngram_min_df=int(feature_config["ngram"]["min_df"])
        )
        oracle_flags = online_true_label_creator_flags(train, target)
        oracle_matrices = {
            name: (train_matrix, replace_creator_column(target_matrix, oracle_flags))
            for name, (train_matrix, target_matrix) in matrices.items()
        }
        frozen_pred, _, _, _ = fit_predict_frozen_mulcas(
            train, target, matrices, model_config, mulcas_config, seed=seed
        )
        oracle_pred, _, _, _ = fit_predict_frozen_mulcas(
            train, target, oracle_matrices, model_config, mulcas_config, seed=seed
        )
        frozen = _prediction_frame(
            frozen_pred, target, "frozen", frozen_flags, oracle_flags
        )
        oracle = _prediction_frame(
            oracle_pred, target, "oracle", frozen_flags, oracle_flags
        )[
            ["row_index", "oracle_prediction", "oracle_score", "oracle_accepted_viewer"]
        ]
        joined = frozen.merge(oracle, on="row_index", validate="one_to_one")
        archived = archived_predictions.loc[
            archived_predictions["period"] == scenario.prediction_period,
            ["row_index", "final_prediction", "final_score"],
        ]
        replay = joined.merge(archived, on="row_index", validate="one_to_one")
        exact = bool(
            np.array_equal(replay["frozen_prediction"], replay["final_prediction"])
            and np.allclose(replay["frozen_score"], replay["final_score"], atol=1e-12)
        )
        replay_checks[scenario.prediction_period] = exact
        if not exact:
            raise ValueError(f"Frozen replay mismatch for {scenario.prediction_period}.")

        for condition in ("frozen", "oracle"):
            metrics = binary_classification_metrics(
                joined["label"].to_numpy(dtype=int),
                joined[f"{condition}_prediction"].to_numpy(dtype=int),
                joined[f"{condition}_score"].to_numpy(dtype=float),
            )
            metric_rows.append(
                {
                    "period": scenario.prediction_period,
                    "condition": condition,
                    "deployability": "deployable_frozen_control"
                    if condition == "frozen"
                    else "non_deployable_true_label_online_oracle",
                    **metrics,
                    "creator_flag_count": int(
                        joined[f"{condition}_creator_flag"].sum()
                        if f"{condition}_creator_flag" in joined
                        else joined["oracle_creator_flag"].sum()
                    ),
                }
            )
        changed = joined.loc[
            (joined["frozen_creator_flag"] != joined["oracle_creator_flag"])
            | (joined["frozen_prediction"] != joined["oracle_prediction"])
        ].copy()
        changed["creator_flag_changed"] = changed["frozen_creator_flag"] != changed["oracle_creator_flag"]
        changed["prediction_changed"] = changed["frozen_prediction"] != changed["oracle_prediction"]
        changed["non_deployable"] = True
        changed_rows.append(changed)

    changed_frame = pd.concat(changed_rows, ignore_index=True) if changed_rows else pd.DataFrame()
    return pd.DataFrame(metric_rows), changed_frame, {
        "frozen_replay_exact_by_period": replay_checks,
        "all_frozen_replays_exact": all(replay_checks.values()),
        "oracle_policy": "predict row before revealing labels; commit true-positive creators after the whole creation block",
    }
