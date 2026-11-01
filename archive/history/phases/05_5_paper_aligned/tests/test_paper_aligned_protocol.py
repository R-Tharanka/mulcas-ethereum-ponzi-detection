"""Leakage and chronology tests for the frozen paper-aligned protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.paper_aligned_split import (
    PonziOccurrenceSplitConfig,
    assign_ponzi_occurrence_splits,
)
from src.features.developer import ChronologicalCreatorTransformer
from src.features.opcode import make_paper_aligned_ngram_tfidf_vectorizer
from src.features.word2vec import IDFWeightedOpcodeWord2VecTransformer
from src.models.run_baselines import validate_evaluation_protocol


def _occurrence_config() -> PonziOccurrenceSplitConfig:
    return PonziOccurrenceSplitConfig(
        train_positive_end=2,
        validation_positive_end=3,
        expected_total_positives=4,
    )


def test_occurrence_split_is_chronological_complete_and_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "address": ["late", "p1", "b1", "p2", "p3", "b2", "p4"],
            "createdBlockNumber": [70, 10, 20, 30, 50, 30, 80],
            "label": [0, 1, 0, 1, 1, 0, 1],
        }
    )
    split, boundaries = assign_ponzi_occurrence_splits(frame, _occurrence_config())

    assert split["createdBlockNumber"].is_monotonic_increasing
    assert split["row_index"].is_unique
    assert set(split["row_index"]) == set(frame.index)
    assert boundaries == {"train_end_block": 30, "validation_end_block": 50}
    assert split.groupby("split")["label"].sum().to_dict() == {
        "test": 1,
        "train": 2,
        "validation": 1,
    }
    assert set(split.loc[split["createdBlockNumber"] == 30, "split"]) == {"train"}


def test_paper_ngram_transform_never_learns_validation_tokens() -> None:
    vectorizer = make_paper_aligned_ngram_tfidf_vectorizer(min_df=2).fit(
        ["PUSH1 ADD", "PUSH1 ADD STOP"]
    )
    vocabulary_before = dict(vectorizer.vocabulary_)
    idf_before = vectorizer.idf_.copy()

    transformed = vectorizer.transform(["FUTURE_ONLY FUTURE_ONLY"])

    assert "FUTURE_ONLY" not in vectorizer.vocabulary_
    assert vectorizer.vocabulary_ == vocabulary_before
    np.testing.assert_array_equal(vectorizer.idf_, idf_before)
    assert transformed.shape[1] == len(vocabulary_before)


def test_idf_word2vec_transform_does_not_learn_future_state() -> None:
    transformer = IDFWeightedOpcodeWord2VecTransformer(
        vector_size=8, epochs=3, workers=1
    ).fit(["PUSH1 ADD", "PUSH1 STOP"])
    vocabulary_before = set(transformer.model_.wv.key_to_index)
    idf_before = dict(transformer.idf_)

    transformed = transformer.transform(["FUTURE_ONLY"])

    assert set(transformer.model_.wv.key_to_index) == vocabulary_before
    assert transformer.idf_ == idf_before
    np.testing.assert_array_equal(transformed, np.zeros((1, 8), dtype=np.float32))


def test_creator_feature_cannot_access_future_labels() -> None:
    training = pd.DataFrame(
        {
            "creator": ["known", "future"],
            "createdBlockNumber": [10, 20],
            "label": [1, 0],
        }
    )
    future_without_labels = pd.DataFrame(
        {
            "creator": ["future", "known"],
            "createdBlockNumber": [30, 40],
        }
    )
    transformer = ChronologicalCreatorTransformer().fit(training)

    values = transformer.transform(future_without_labels)

    np.testing.assert_array_equal(values[:, 0], [0.0, 1.0])


def test_final_test_cannot_be_configured_for_selection() -> None:
    validate_evaluation_protocol(
        {"selection_split": "validation", "final_test_usage": "reporting_only"}
    )
    with pytest.raises(ValueError, match="validation"):
        validate_evaluation_protocol(
            {"selection_split": "test", "final_test_usage": "reporting_only"}
        )
    with pytest.raises(ValueError, match="reporting only"):
        validate_evaluation_protocol(
            {"selection_split": "validation", "final_test_usage": "threshold_tuning"}
        )
