"""Label-independent N-gram dimensionality audit for Phase 9A."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.opcode import make_paper_aligned_ngram_count_vectorizer


PAPER_DIMENSIONS = {
    (1, 1): (141, 77),
    (1, 2): (7211, 767),
    (1, 3): (52532, 2292),
}


def audit_ngram_dimensions(
    corpora: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Measure observed dimensions and integer document-frequency cutoffs."""
    rows: list[dict[str, Any]] = []
    histograms: dict[str, dict[str, int]] = {}
    for corpus_name, frame in corpora.items():
        documents = frame["opcode"].astype(str).tolist()
        for ngram_range, (paper_raw, paper_selected) in PAPER_DIMENSIONS.items():
            vectorizer = make_paper_aligned_ngram_count_vectorizer(1)
            vectorizer.set_params(ngram_range=ngram_range)
            matrix = vectorizer.fit_transform(documents).tocsr()
            document_frequency = np.asarray((matrix > 0).sum(axis=0)).reshape(-1)
            unique_df, counts = np.unique(document_frequency, return_counts=True)
            key = f"{corpus_name}__{ngram_range[0]}_{ngram_range[1]}"
            histograms[key] = {
                str(int(df)): int(count) for df, count in zip(unique_df, counts, strict=True)
            }
            cutoff_dimensions = {
                cutoff: int((document_frequency >= cutoff).sum())
                for cutoff in range(1, int(document_frequency.max()) + 1)
            }
            nearest_cutoff = min(
                cutoff_dimensions,
                key=lambda cutoff: (abs(cutoff_dimensions[cutoff] - paper_selected), cutoff),
            )
            exact_cutoffs = [
                cutoff for cutoff, dimension in cutoff_dimensions.items()
                if dimension == paper_selected
            ]
            rows.append(
                {
                    "corpus": corpus_name,
                    "ngram_range": f"{ngram_range[0]}-{ngram_range[1]}",
                    "row_count": len(frame),
                    "paper_before_selection": paper_raw,
                    "observed_before_selection": int(matrix.shape[1]),
                    "raw_dimension_status": "MATCH" if matrix.shape[1] == paper_raw else "MISMATCH",
                    "paper_after_selection": paper_selected,
                    "observed_min_df_2": int((document_frequency >= 2).sum()),
                    "exact_integer_cutoffs": ";".join(map(str, exact_cutoffs)),
                    "nearest_integer_cutoff": nearest_cutoff,
                    "nearest_dimension": cutoff_dimensions[nearest_cutoff],
                    "nearest_dimension_delta": cutoff_dimensions[nearest_cutoff] - paper_selected,
                    "paper_df_wording": "discard document frequency less than 0.1; formula semantics under-specified",
                    "classifier_trained": False,
                }
            )
    return pd.DataFrame(rows), histograms
