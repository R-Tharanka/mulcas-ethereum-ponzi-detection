"""Frozen opcode-encoder inference and deterministic embedding caches."""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.experiments.phase6a import sha256_file
from src.extension.sequence_data import OpcodeVocabulary, fixed_size_chunks
from src.extension.sequence_model import OpcodeMaskedLanguageModel
from src.extension.sequence_pretraining import encoder_hidden_states, masked_mean_pool


@dataclass(frozen=True)
class LoadedEncoder:
    model: OpcodeMaskedLanguageModel
    vocabulary: OpcodeVocabulary
    metadata: dict[str, Any]
    checkpoint_hash: str


def load_frozen_encoder(
    checkpoint_entry: dict[str, Any], *, device: torch.device
) -> LoadedEncoder:
    path = Path(checkpoint_entry["path"])
    observed_hash = sha256_file(path)
    if observed_hash != checkpoint_entry["sha256"]:
        raise RuntimeError(f"Frozen checkpoint hash mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    architecture = metadata["architecture"]
    if metadata["fold"] != checkpoint_entry["fold"] or int(metadata["seed"]) != int(checkpoint_entry["seed"]):
        raise RuntimeError("Checkpoint fold/seed metadata does not match its manifest.")
    if metadata["vocabulary_hash"] != checkpoint_entry["vocabulary_hash"]:
        raise RuntimeError("Checkpoint vocabulary hash does not match its manifest.")
    model = OpcodeMaskedLanguageModel(
        len(metadata["vocabulary"]),
        maximum_length=int(metadata["sequence_policy"]["length"]),
        embedding_dim=int(architecture["vocabulary_embedding_dim"]),
        encoder_layers=int(architecture["encoder_layers"]),
        attention_heads=int(architecture["attention_heads"]),
        feed_forward_dim=int(architecture["feed_forward_dim"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return LoadedEncoder(
        model=model,
        vocabulary=OpcodeVocabulary({str(key): int(value) for key, value in metadata["vocabulary"].items()}),
        metadata=metadata,
        checkpoint_hash=observed_hash,
    )


def prepare_contract_chunks(
    frame: pd.DataFrame,
    vocabulary: OpcodeVocabulary,
    *,
    chunk_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return padded chunks, owner positions, contract rows, and chunk counts."""
    chunks: list[list[int]] = []
    owners: list[int] = []
    chunk_counts = np.zeros(len(frame), dtype=np.int64)
    for owner, tokens in enumerate(frame["tokens"]):
        contract_chunks = fixed_size_chunks(tokens, chunk_length)
        chunk_counts[owner] = len(contract_chunks)
        for chunk in contract_chunks:
            encoded = vocabulary.encode(chunk)
            chunks.append(encoded + [vocabulary.pad_id] * (chunk_length - len(encoded)))
            owners.append(owner)
    matrix = np.asarray(chunks, dtype=np.uint16)
    if matrix.shape != (len(chunks), chunk_length):
        raise AssertionError("Frozen chunk matrix has an unexpected shape.")
    if int(chunk_counts.sum()) != len(chunks) or bool((chunk_counts < 1).any()):
        raise AssertionError("Every contract must own at least one chunk exactly once.")
    return (
        matrix,
        np.asarray(owners, dtype=np.int64),
        frame["row_index"].to_numpy(dtype=np.int64),
        chunk_counts,
    )


def extract_contract_embeddings(
    loaded: LoadedEncoder,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    amp: bool,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract contextual and embedding-table controls in one frozen pass."""
    chunk_length = int(loaded.metadata["sequence_policy"]["length"])
    chunks, owners, row_indices, chunk_counts = prepare_contract_chunks(
        frame, loaded.vocabulary, chunk_length=chunk_length
    )
    dimension = int(loaded.metadata["architecture"]["vocabulary_embedding_dim"])
    contextual_sum = np.zeros((len(frame), dimension), dtype=np.float64)
    non_contextual_sum = np.zeros((len(frame), dimension), dtype=np.float64)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            selected = slice(start, start + batch_size)
            token_ids = torch.as_tensor(
                chunks[selected].astype(np.int64, copy=False), device=device, dtype=torch.long
            )
            padding_mask = token_ids.eq(loaded.vocabulary.pad_id)
            valid_rows = (~padding_mask).any(dim=1)
            contextual = torch.zeros(
                (len(token_ids), dimension), device=device, dtype=torch.float32
            )
            non_contextual = torch.zeros_like(contextual)
            if bool(valid_rows.any()):
                valid_ids = token_ids[valid_rows]
                valid_padding = padding_mask[valid_rows]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    token_embeddings = loaded.model.embedding(valid_ids)
                    hidden_states = encoder_hidden_states(loaded.model, valid_ids, valid_padding)
                    non_contextual[valid_rows] = masked_mean_pool(token_embeddings, valid_padding).float()
                    contextual[valid_rows] = masked_mean_pool(hidden_states, valid_padding).float()
            owner_batch = owners[selected]
            np.add.at(contextual_sum, owner_batch, contextual.cpu().numpy())
            np.add.at(non_contextual_sum, owner_batch, non_contextual.cpu().numpy())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    contextual_result = (contextual_sum / chunk_counts[:, None]).astype(np.float32)
    non_contextual_result = (non_contextual_sum / chunk_counts[:, None]).astype(np.float32)
    if not np.isfinite(contextual_result).all() or not np.isfinite(non_contextual_result).all():
        raise FloatingPointError("Frozen embedding extraction produced non-finite values.")
    diagnostics = {
        "contracts": int(len(frame)),
        "chunks": int(len(chunks)),
        "non_padding_tokens": int((chunks != loaded.vocabulary.pad_id).sum()),
        "elapsed_seconds": elapsed,
        "contracts_per_second": len(frame) / elapsed,
        "chunks_per_second": len(chunks) / elapsed,
        "tokens_per_second": int((chunks != loaded.vocabulary.pad_id).sum()) / elapsed,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "gradients_enabled": False,
        "model_parameters_trainable": any(parameter.requires_grad for parameter in loaded.model.parameters()),
        "chunk_count_min": int(chunk_counts.min()),
        "chunk_count_max": int(chunk_counts.max()),
        "chunk_count_sum": int(chunk_counts.sum()),
    }
    return {
        "row_index": row_indices,
        "contextual": contextual_result,
        "non_contextual": non_contextual_result,
        "chunk_count": chunk_counts,
    }, diagnostics


def write_deterministic_embedding_cache(
    path: Path, *, row_index: np.ndarray, embeddings: np.ndarray
) -> str:
    """Write a deterministic compressed NPZ with fixed ZIP metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"row_index": np.asarray(row_index, dtype=np.int64), "embeddings": np.asarray(embeddings, dtype=np.float32)}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return sha256_file(path)


def load_embedding_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        rows = np.asarray(artifact["row_index"], dtype=np.int64)
        embeddings = np.asarray(artifact["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape != (len(rows), 128):
        raise AssertionError("Frozen embedding cache must contain one 128-dimensional row per contract.")
    return rows, embeddings


def checkpoint_entries(path: Path) -> list[dict[str, Any]]:
    entries = json.loads(path.read_text(encoding="utf-8"))["checkpoints"]
    if len(entries) != 9:
        raise AssertionError("Expected exactly nine frozen Phase 8R5B checkpoints.")
    for entry in entries:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError(f"Frozen checkpoint changed: {entry['path']}")
    return entries
