"""Strictly historical opcode-family similarity features for Phase 9C."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

from src.features.opcode import make_paper_aligned_ngram_tfidf_vectorizer


FAMILY_COLUMNS = [
    "max_similarity_prior_ponzi",
    "max_similarity_prior_benign",
    "similarity_margin",
    "nearest_prior_label",
    "count_prior_ponzi_similarity_ge_0_90",
    "count_prior_all_similarity_ge_0_90",
]


@dataclass(frozen=True)
class FamilyStateResult:
    train_raw: pd.DataFrame
    target_raw: pd.DataFrame
    train_model: np.ndarray
    target_model: np.ndarray
    vocabulary_size: int


def transform_family_features(raw: pd.DataFrame) -> np.ndarray:
    values = raw.loc[:, FAMILY_COLUMNS].astype(float).copy()
    values["count_prior_ponzi_similarity_ge_0_90"] = np.log1p(
        values["count_prior_ponzi_similarity_ge_0_90"]
    )
    values["count_prior_all_similarity_ge_0_90"] = np.log1p(
        values["count_prior_all_similarity_ge_0_90"]
    )
    return values.to_numpy(dtype=np.float32)


def _labeled_stats(similarities: np.ndarray, labels: np.ndarray, threshold: float) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    positive = labels == 1
    benign = labels == 0
    for row in similarities:
        max_positive = float(row[positive].max()) if positive.any() else 0.0
        max_benign = float(row[benign].max()) if benign.any() else 0.0
        if not len(labels):
            nearest = -1
        else:
            nearest = 1 if max_positive > max_benign else 0
        output.append({
            "max_similarity_prior_ponzi": max_positive,
            "max_similarity_prior_benign": max_benign,
            "similarity_margin": max_positive - max_benign,
            "nearest_prior_label": nearest,
            "count_prior_ponzi_similarity_ge_0_90": int(
                np.count_nonzero(row[positive] >= threshold)
            ) if positive.any() else 0,
        })
    return output


def _dense_similarity(query: csr_matrix, reference: csr_matrix) -> np.ndarray:
    if reference.shape[0] == 0:
        return np.zeros((query.shape[0], 0), dtype=np.float32)
    return (query @ reference.T).toarray().astype(np.float32, copy=False)


def build_family_state(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    min_df: int,
    threshold: float,
) -> FamilyStateResult:
    """Build labeled-training and label-free prequential family state."""
    if threshold != 0.90:
        raise ValueError("Phase 9C family similarity threshold must remain 0.90.")
    required_train = {"row_index", "opcode", "createdBlockNumber", "label"}
    required_target = {"row_index", "opcode", "createdBlockNumber"}
    if not required_train.issubset(train.columns) or not required_target.issubset(target.columns):
        raise ValueError("Family-state inputs are missing required columns.")

    vectorizer = make_paper_aligned_ngram_tfidf_vectorizer(min_df)
    train_matrix = vectorizer.fit_transform(train["opcode"].astype(str)).tocsr()
    target_matrix = vectorizer.transform(target["opcode"].astype(str)).tocsr()
    train_labels = train["label"].to_numpy(dtype=int)
    train_blocks = train["createdBlockNumber"].to_numpy(dtype=int)
    target_blocks = target["createdBlockNumber"].to_numpy(dtype=int)
    train_opcodes = train["opcode"].astype(str).tolist()
    target_opcodes = target["opcode"].astype(str).tolist()

    chunk_size = 64
    first_train_block: dict[str, int] = {}
    for opcode, block in zip(train_opcodes, train_blocks, strict=True):
        first_train_block[opcode] = min(first_train_block.get(opcode, int(block)), int(block))

    train_records: list[dict[str, float | int | bool]] = [dict() for _ in range(len(train))]
    for start in range(0, len(train), chunk_size):
        end = min(start + chunk_size, len(train))
        similarities = _dense_similarity(train_matrix[start:end], train_matrix[:end])
        for local, position in enumerate(range(start, end)):
            valid = train_blocks[:end] < train_blocks[position]
            row = similarities[local, valid]
            labels = train_labels[:end][valid]
            record = _labeled_stats(row.reshape(1, -1), labels, threshold)[0]
            record["count_prior_all_similarity_ge_0_90"] = int(np.count_nonzero(row >= threshold))
            record["historically_exact_cloned_code"] = first_train_block[train_opcodes[position]] < train_blocks[position]
            train_records[position] = record

    first_all_block = dict(first_train_block)
    for opcode, block in zip(target_opcodes, target_blocks, strict=True):
        first_all_block[opcode] = min(first_all_block.get(opcode, int(block)), int(block))
    target_records: list[dict[str, float | int | bool]] = [dict() for _ in range(len(target))]
    combined = vstack([train_matrix, target_matrix], format="csr")
    combined_blocks = np.concatenate([train_blocks, target_blocks])
    for start in range(0, len(target), chunk_size):
        end = min(start + chunk_size, len(target))
        labeled_similarity = _dense_similarity(target_matrix[start:end], train_matrix)
        labeled = _labeled_stats(labeled_similarity, train_labels, threshold)
        all_end = len(train) + end
        all_similarity = _dense_similarity(target_matrix[start:end], combined[:all_end])
        for local, position in enumerate(range(start, end)):
            valid = combined_blocks[:all_end] < target_blocks[position]
            row = all_similarity[local, valid]
            record = labeled[local]
            record["count_prior_all_similarity_ge_0_90"] = int(np.count_nonzero(row >= threshold))
            record["historically_exact_cloned_code"] = first_all_block[target_opcodes[position]] < target_blocks[position]
            target_records[position] = record

    train_raw = pd.DataFrame(train_records)
    train_raw.insert(0, "row_index", train["row_index"].to_numpy(dtype=int))
    target_raw = pd.DataFrame(target_records)
    target_raw.insert(0, "row_index", target["row_index"].to_numpy(dtype=int))
    return FamilyStateResult(
        train_raw=train_raw,
        target_raw=target_raw,
        train_model=transform_family_features(train_raw),
        target_model=transform_family_features(target_raw),
        vocabulary_size=len(vectorizer.vocabulary_),
    )
