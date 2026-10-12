from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.extension.phase8r5c1 import load_config
from src.extension.sequence_embeddings import (
    checkpoint_entries,
    extract_contract_embeddings,
    load_embedding_cache,
    load_frozen_encoder,
    prepare_contract_chunks,
    write_deterministic_embedding_cache,
)


def test_prepare_chunks_preserves_every_chunk_and_contract() -> None:
    config = load_config()
    entry = checkpoint_entries(config["inputs"]["checkpoint_manifest"])[0]
    loaded = load_frozen_encoder(entry, device=torch.device("cpu"))
    frame = pd.DataFrame({
        "row_index": [10, 11, 12],
        "tokens": [("ADD",), tuple("ADD" for _ in range(1025)), ()],
    })
    chunks, owners, rows, counts = prepare_contract_chunks(frame, loaded.vocabulary, chunk_length=1024)
    assert chunks.shape == (4, 1024)
    assert counts.tolist() == [1, 2, 1]
    assert np.bincount(owners, minlength=3).tolist() == counts.tolist()
    assert rows.tolist() == [10, 11, 12]


def test_frozen_encoder_has_no_trainable_parameters() -> None:
    config = load_config()
    entry = checkpoint_entries(config["inputs"]["checkpoint_manifest"])[0]
    loaded = load_frozen_encoder(entry, device=torch.device("cpu"))
    assert loaded.model.training is False
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())


def test_embedding_cache_is_hash_deterministic(tmp_path) -> None:
    rows = np.asarray([1, 2], dtype=np.int64)
    embeddings = np.arange(256, dtype=np.float32).reshape(2, 128)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_hash = write_deterministic_embedding_cache(first, row_index=rows, embeddings=embeddings)
    second_hash = write_deterministic_embedding_cache(second, row_index=rows, embeddings=embeddings)
    assert first_hash == second_hash
    observed_rows, observed_embeddings = load_embedding_cache(first)
    np.testing.assert_array_equal(observed_rows, rows)
    np.testing.assert_array_equal(observed_embeddings, embeddings)


def test_cuda_extraction_returns_one_finite_128d_embedding_per_contract() -> None:
    config = load_config()
    entry = checkpoint_entries(config["inputs"]["checkpoint_manifest"])[0]
    loaded = load_frozen_encoder(entry, device=torch.device("cuda:0"))
    frame = pd.DataFrame({
        "row_index": [0, 1],
        "tokens": [("ADD", "STOP"), ()],
    })
    with torch.enable_grad():
        extracted, diagnostics = extract_contract_embeddings(
            loaded, frame, batch_size=2, amp=True, device=torch.device("cuda:0")
        )
    assert extracted["contextual"].shape == (2, 128)
    assert extracted["non_contextual"].shape == (2, 128)
    assert np.isfinite(extracted["contextual"]).all()
    assert np.isfinite(extracted["non_contextual"]).all()
    np.testing.assert_array_equal(extracted["contextual"][1], np.zeros(128, dtype=np.float32))
    assert diagnostics["gradients_enabled"] is False
    assert diagnostics["model_parameters_trainable"] is False
    assert diagnostics["chunk_count_sum"] == 2
