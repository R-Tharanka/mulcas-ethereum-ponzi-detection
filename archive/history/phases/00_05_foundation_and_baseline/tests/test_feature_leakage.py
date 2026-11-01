"""Targeted tests for feature-view temporal leakage controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.developer import ChronologicalCreatorTransformer
from src.features.opcode import (
    make_opcode_tfidf_vectorizer,
    transform_documents,
)
from src.features.word2vec import OpcodeWord2VecTransformer


def test_validation_tokens_never_enter_training_vocabulary() -> None:
    vectorizer = make_opcode_tfidf_vectorizer().fit(["PUSH1 ADD", "PUSH1 STOP"])
    vocabulary_before = dict(vectorizer.vocabulary_)
    idf_before = vectorizer.idf_.copy()

    validation = transform_documents(vectorizer, ["FUTURE_ONLY PUSH1"])

    assert "FUTURE_ONLY" not in vectorizer.vocabulary_
    assert vectorizer.vocabulary_ == vocabulary_before
    np.testing.assert_array_equal(vectorizer.idf_, idf_before)
    assert validation.shape[1] == len(vocabulary_before)


def test_word2vec_transform_does_not_learn_future_tokens() -> None:
    transformer = OpcodeWord2VecTransformer(vector_size=8, epochs=3, workers=1).fit(
        ["PUSH1 ADD", "PUSH1 STOP"]
    )
    vocabulary_before = set(transformer.model_.wv.key_to_index)

    transformed = transformer.transform(["FUTURE_ONLY"])

    assert set(transformer.model_.wv.key_to_index) == vocabulary_before
    np.testing.assert_array_equal(transformed, np.zeros((1, 8), dtype=np.float32))


def test_future_positive_label_cannot_affect_earlier_creator_feature() -> None:
    frame = pd.DataFrame(
        {
            "creator": ["creator-a", "creator-a", "creator-a"],
            "createdBlockNumber": [10, 20, 30],
            "label": [0, 0, 1],
        }
    )

    features = ChronologicalCreatorTransformer().fit_transform(frame)

    np.testing.assert_array_equal(features[:, 0], [0.0, 0.0, 0.0])


def test_same_block_positive_label_cannot_leak_to_peer_contract() -> None:
    frame = pd.DataFrame(
        {
            "creator": ["creator-a", "creator-a", "creator-a"],
            "createdBlockNumber": [10, 10, 20],
            "label": [1, 0, 0],
        }
    )

    features = ChronologicalCreatorTransformer().fit_transform(frame)

    np.testing.assert_array_equal(features[:, 0], [0.0, 0.0, 1.0])


def test_validation_labels_are_not_required_or_used_during_transform() -> None:
    train = pd.DataFrame(
        {
            "creator": ["known-positive", "other"],
            "createdBlockNumber": [10, 20],
            "label": [1, 0],
        }
    )
    validation_without_labels = pd.DataFrame(
        {
            "creator": ["known-positive", "future-positive"],
            "createdBlockNumber": [30, 40],
        }
    )
    transformer = ChronologicalCreatorTransformer().fit(train)

    features = transformer.transform(validation_without_labels)

    np.testing.assert_array_equal(features[:, 0], [1.0, 0.0])
    assert "future-positive" not in transformer.positive_creators_
