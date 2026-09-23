"""Training-only opcode count and TF-IDF feature views."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def tokenize_opcodes(document: str) -> list[str]:
    """Tokenize a whitespace-separated opcode document."""
    return document.split()


def _common_vectorizer_args() -> dict[str, object]:
    return {
        "tokenizer": tokenize_opcodes,
        "token_pattern": None,
        "preprocessor": None,
        "lowercase": False,
        "dtype": np.float32,
    }


def make_opcode_term_count_vectorizer() -> CountVectorizer:
    """Create the published unigram opcode term-count view."""
    return CountVectorizer(ngram_range=(1, 1), **_common_vectorizer_args())


def make_opcode_tfidf_vectorizer() -> TfidfVectorizer:
    """Create the published unigram opcode TF-IDF view."""
    return TfidfVectorizer(ngram_range=(1, 1), **_common_vectorizer_args())


def make_opcode_ngram_count_vectorizer() -> CountVectorizer:
    """Create the opcode unigram/bigram/trigram count view."""
    return CountVectorizer(ngram_range=(1, 3), **_common_vectorizer_args())


def make_opcode_ngram_tfidf_vectorizer() -> TfidfVectorizer:
    """Create the opcode unigram/bigram/trigram TF-IDF view."""
    return TfidfVectorizer(ngram_range=(1, 3), **_common_vectorizer_args())


def fit_training_transformer(
    vectorizer: CountVectorizer | TfidfVectorizer,
    train_documents: Sequence[str],
) -> CountVectorizer | TfidfVectorizer:
    """Fit a vectorizer on training documents only."""
    vectorizer.fit(train_documents)
    return vectorizer


def transform_documents(
    vectorizer: CountVectorizer | TfidfVectorizer,
    documents: Sequence[str],
) -> csr_matrix:
    """Transform documents without changing fitted vocabulary or IDF state."""
    return vectorizer.transform(documents).tocsr()
