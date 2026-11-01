"""Deterministic training-only Word2Vec contract representations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.opcode import _common_vectorizer_args, tokenize_opcodes


@dataclass
class OpcodeWord2VecTransformer:
    """Represent a contract by the mean of its in-vocabulary opcode vectors."""

    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    epochs: int = 10
    seed: int = 42
    workers: int = 1
    model_: Word2Vec | None = field(default=None, init=False, repr=False)

    def fit(self, documents: Sequence[str]) -> "OpcodeWord2VecTransformer":
        """Fit Word2Vec using training documents only."""
        sentences = [tokenize_opcodes(document) for document in documents]
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
            epochs=self.epochs,
            sorted_vocab=1,
        )
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Transform documents without updating the fitted embedding model."""
        if self.model_ is None:
            raise RuntimeError("OpcodeWord2VecTransformer must be fitted first.")

        matrix = np.zeros((len(documents), self.vector_size), dtype=np.float32)
        vocabulary = self.model_.wv.key_to_index
        for row_index, document in enumerate(documents):
            tokens = [token for token in tokenize_opcodes(document) if token in vocabulary]
            if tokens:
                matrix[row_index] = np.mean(self.model_.wv[tokens], axis=0)
        return matrix

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        """Fit on training documents and return their representations."""
        return self.fit(documents).transform(documents)

    @property
    def vocabulary_size_(self) -> int:
        """Return the fitted vocabulary size."""
        if self.model_ is None:
            raise RuntimeError("OpcodeWord2VecTransformer must be fitted first.")
        return len(self.model_.wv)


@dataclass
class IDFWeightedOpcodeWord2VecTransformer:
    """Represent contracts by the IDF-weighted sum of opcode embeddings."""

    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    epochs: int = 10
    seed: int = 42
    workers: int = 1
    model_: Word2Vec | None = field(default=None, init=False, repr=False)
    idf_: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def fit(self, documents: Sequence[str]) -> "IDFWeightedOpcodeWord2VecTransformer":
        """Fit embeddings and IDF statistics using training documents only."""
        sentences = [tokenize_opcodes(document) for document in documents]
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
            epochs=self.epochs,
            sorted_vocab=1,
        )
        idf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            **_common_vectorizer_args(),
        ).fit(documents)
        self.idf_ = dict(
            zip(
                idf_vectorizer.get_feature_names_out(),
                idf_vectorizer.idf_,
                strict=True,
            )
        )
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Transform documents without updating embeddings or IDF statistics."""
        if self.model_ is None or not self.idf_:
            raise RuntimeError("IDFWeightedOpcodeWord2VecTransformer must be fitted first.")
        matrix = np.zeros((len(documents), self.vector_size), dtype=np.float32)
        vocabulary = self.model_.wv.key_to_index
        for row_index, document in enumerate(documents):
            tokens = [
                token
                for token in tokenize_opcodes(document)
                if token in vocabulary and token in self.idf_
            ]
            if tokens:
                weights = np.asarray([self.idf_[token] for token in tokens], dtype=np.float32)
                vectors = self.model_.wv[tokens]
                matrix[row_index] = (vectors * weights[:, None]).sum(axis=0)
        return matrix

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    @property
    def vocabulary_size_(self) -> int:
        if self.model_ is None:
            raise RuntimeError("IDFWeightedOpcodeWord2VecTransformer must be fitted first.")
        return len(self.model_.wv)
