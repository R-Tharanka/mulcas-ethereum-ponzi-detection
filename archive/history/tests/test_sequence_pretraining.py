from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.extension.sequence_data import SPECIAL_TOKENS
from src.extension.sequence_model import OpcodeMaskedLanguageModel
from src.extension.sequence_pretraining import (
    apply_mlm_mask,
    encoder_hidden_states,
    masked_mean_pool,
    prepare_fold_corpus,
)


def _small_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "row_index": [0, 1, 2],
        "period": ["P0", "P1", "P2"],
        "tokens": [("ADD", "STOP"), ("PUSH1", "RETURN"), ("FUTURE_ONLY",)],
    })


def test_fold_vocabulary_and_chunks_use_historical_rows_only() -> None:
    corpus = prepare_fold_corpus(
        _small_frame(), fold="P2", historical_periods=("P0", "P1"), chunk_length=4
    )
    assert corpus.row_indices == (0, 1)
    assert "FUTURE_ONLY" not in corpus.vocabulary.token_to_id
    assert corpus.chunks.shape == (2, 4)
    assert corpus.non_padding_tokens == 4
    assert corpus.padding_tokens == 4


def test_chunking_preserves_all_non_padding_tokens() -> None:
    frame = pd.DataFrame({
        "row_index": [0], "period": ["P0"],
        "tokens": [tuple(f"OP{index}" for index in range(10))],
    })
    corpus = prepare_fold_corpus(frame, fold="P2", historical_periods=("P0",), chunk_length=4)
    assert corpus.chunks.shape == (3, 4)
    assert corpus.non_padding_tokens == 10
    assert corpus.padding_tokens == 2


def test_masking_is_deterministic_protects_specials_and_uses_80_10_10() -> None:
    ordinary = torch.arange(3, 103, dtype=torch.long).repeat(100, 1)
    specials = torch.tensor([[0, 1, 2]], dtype=torch.long).repeat(100, 1)
    tokens = torch.cat([specials, ordinary], dim=1)
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)
    first = apply_mlm_mask(
        tokens, vocabulary_size=103, mask_id=1,
        protected_token_count=len(SPECIAL_TOKENS), probability=0.15,
        generator=first_generator,
    )
    second = apply_mlm_mask(
        tokens, vocabulary_size=103, mask_id=1,
        protected_token_count=len(SPECIAL_TOKENS), probability=0.15,
        generator=second_generator,
    )
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    assert not bool(first[2][:, :3].any())
    counts = first[3]
    total = counts["selected"]
    assert total == counts["mask_replaced"] + counts["random_replaced"] + counts["unchanged"]
    assert 0.75 < counts["mask_replaced"] / total < 0.85
    assert 0.07 < counts["random_replaced"] / total < 0.13
    assert 0.07 < counts["unchanged"] / total < 0.13


def test_masked_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    padding = torch.tensor([[False, False, True]])
    torch.testing.assert_close(masked_mean_pool(hidden, padding), torch.tensor([[2.0, 3.0]]))


def test_encoder_api_exposes_token_and_chunk_representations() -> None:
    model = OpcodeMaskedLanguageModel(12, maximum_length=4).eval()
    tokens = torch.tensor([[3, 4, 0, 0]])
    padding = tokens.eq(0)
    with torch.no_grad():
        hidden = encoder_hidden_states(model, tokens, padding)
        pooled = masked_mean_pool(hidden, padding)
    assert hidden.shape == (1, 4, 128)
    assert pooled.shape == (1, 128)
    assert np.isfinite(pooled.numpy()).all()
