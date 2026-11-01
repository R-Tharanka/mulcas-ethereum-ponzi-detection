"""Fixed-candidate, non-deployable P5 viewer-selection oracle."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.evaluation.metrics import binary_classification_metrics
from src.features.developer import ChronologicalCreatorTransformer
from src.features.opcode import (
    make_opcode_term_count_vectorizer,
    make_opcode_tfidf_vectorizer,
    make_paper_aligned_ngram_count_vectorizer,
    make_paper_aligned_ngram_tfidf_vectorizer,
)
from src.features.paper_aligned import _word2vec_from_config
from src.models.baselines import build_baseline_model, positive_class_scores


VIEW_ORDER = (
    "opcode_term_count",
    "opcode_tfidf",
    "opcode_ngram_count_1_3_df_filtered",
    "opcode_ngram_tfidf_1_3_df_filtered",
    "opcode_word2vec_idf_weighted",
    "opcode_word2vec_mean",
)
MODEL_ORDER = ("ridge", "svm", "xgboost")


def build_candidate_matrices(
    train: pd.DataFrame,
    target: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[dict[str, tuple[Any, Any]], float]:
    """Fit only the six frozen Phase 5.5 views on P0-P4."""
    started = time.perf_counter()
    train_docs = train["opcode"].astype(str).tolist()
    target_docs = target["opcode"].astype(str).tolist()
    creator = ChronologicalCreatorTransformer()
    train_creator = creator.fit_transform(train).reshape(-1, 1)
    target_creator = creator.transform(target[["creator", "createdBlockNumber"]]).reshape(-1, 1)
    min_df = int(feature_config["ngram"]["min_df"])
    vectorizers = {
        "opcode_term_count": make_opcode_term_count_vectorizer(),
        "opcode_tfidf": make_opcode_tfidf_vectorizer(),
        "opcode_ngram_count_1_3_df_filtered": make_paper_aligned_ngram_count_vectorizer(min_df),
        "opcode_ngram_tfidf_1_3_df_filtered": make_paper_aligned_ngram_tfidf_vectorizer(min_df),
    }
    matrices: dict[str, tuple[Any, Any]] = {}
    for name, transformer in vectorizers.items():
        train_static = transformer.fit_transform(train_docs).tocsr()
        target_static = transformer.transform(target_docs).tocsr()
        matrices[name] = (
            hstack([train_static, csr_matrix(train_creator)], format="csr"),
            hstack([target_static, csr_matrix(target_creator)], format="csr"),
        )
    for name, weighted in (
        ("opcode_word2vec_idf_weighted", True),
        ("opcode_word2vec_mean", False),
    ):
        transformer = _word2vec_from_config(feature_config, weighted).fit(train_docs)
        matrices[name] = (
            np.column_stack([transformer.transform(train_docs), train_creator]),
            np.column_stack([transformer.transform(target_docs), target_creator]),
        )
    return matrices, time.perf_counter() - started


def run_viewer_selection_oracle(
    train: pd.DataFrame,
    target: pd.DataFrame,
    feature_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    seed: int,
    runtime_gate_minutes: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate the fixed 18-candidate pool on observed P5 labels."""
    matrices, feature_seconds = build_candidate_matrices(train, target, feature_config)
    labels_train = train["label"].to_numpy(dtype=int)
    labels_target = target["label"].to_numpy(dtype=int)

    representative_view = "opcode_ngram_count_1_3_df_filtered"
    representative_model = "xgboost"
    representative = build_baseline_model(
        representative_model, model_config["models"][representative_model], seed
    )
    preflight_started = time.perf_counter()
    representative.fit(matrices[representative_view][0], labels_train)
    representative_seconds = time.perf_counter() - preflight_started
    estimated_seconds = feature_seconds + representative_seconds * len(VIEW_ORDER) * len(MODEL_ORDER)
    preflight = {
        "feature_construction_seconds": feature_seconds,
        "representative_candidate": f"{representative_model}__{representative_view}",
        "representative_fit_seconds": representative_seconds,
        "candidate_count": len(VIEW_ORDER) * len(MODEL_ORDER),
        "estimated_total_seconds": estimated_seconds,
        "runtime_gate_seconds": runtime_gate_minutes * 60.0,
        "gate_passed": estimated_seconds <= runtime_gate_minutes * 60.0,
    }
    if not preflight["gate_passed"]:
        return pd.DataFrame(), pd.DataFrame(), preflight

    rows: list[dict[str, Any]] = []
    candidate_order = 0
    for view in VIEW_ORDER:
        x_train, x_target = matrices[view]
        for family in MODEL_ORDER:
            candidate_order += 1
            model = build_baseline_model(family, model_config["models"][family], seed)
            started = time.perf_counter()
            model.fit(x_train, labels_train)
            fit_seconds = time.perf_counter() - started
            predictions = np.asarray(model.predict(x_target), dtype=int)
            scores = positive_class_scores(model, x_target)
            metrics = binary_classification_metrics(labels_target, predictions, scores)
            rows.append(
                {
                    "candidate_id": f"{family}__{view}",
                    "model_family": family,
                    "feature_view": view,
                    "candidate_order": candidate_order,
                    "selection_data": "P5_true_labels_non_deployable_oracle",
                    "fit_periods": "P0-P4",
                    "evaluation_period": "P5",
                    "fit_seconds": fit_seconds,
                    **metrics,
                }
            )
    metrics_frame = pd.DataFrame(rows)
    ranked = metrics_frame.sort_values(
        ["f1", "precision", "recall", "candidate_order"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "oracle_rank", np.arange(1, len(ranked) + 1))
    ranked["paper_says_five_final_models"] = ranked["oracle_rank"] <= 5
    ranked["full_mulcas_replay_status"] = "not_reproducible_from_available_specification"
    ranked["reason"] = (
        "Paper states five final models but enumerates four and does not specify a unique fifth viewer or complete cascade ordering."
    )
    preflight["actual_candidate_fit_seconds"] = float(metrics_frame["fit_seconds"].sum())
    preflight["full_replay_status"] = "not_reproducible_from_available_specification"
    return metrics_frame, ranked, preflight
