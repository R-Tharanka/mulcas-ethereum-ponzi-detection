"""Reusable, temporally safe masked-opcode pretraining utilities."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.extension.sequence_data import (
    SPECIAL_TOKENS,
    OpcodeVocabulary,
    build_vocabulary,
    canonical_opcode_tokens,
    fixed_size_chunks,
)
from src.extension.sequence_model import OpcodeMaskedLanguageModel, masked_opcode_loss
from src.utils.seed import set_random_seed


EXPECTED_PERIOD_ROWS = {"P0": 104, "P1": 305, "P2": 5141, "P3": 289, "P4": 150}
P0_P4_ROWS = sum(EXPECTED_PERIOD_ROWS.values())


@dataclass(frozen=True)
class PreparedCorpus:
    fold: str
    historical_periods: tuple[str, ...]
    row_indices: tuple[int, ...]
    vocabulary: OpcodeVocabulary
    vocabulary_hash: str
    chunks: np.ndarray
    contract_count: int
    non_padding_tokens: int
    padding_tokens: int
    mean_chunks_per_contract: float
    max_chunks_per_contract: int


def vocabulary_sha256(vocabulary: OpcodeVocabulary) -> str:
    payload = json.dumps(
        vocabulary.token_to_id, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_p0_p4_sequences(dataset: Path, period_definitions: Path) -> pd.DataFrame:
    """Load only label-free P0-P4 rows and attach frozen period membership."""
    frame = pd.read_csv(
        dataset,
        usecols=["createdBlockNumber", "opcode"],
        nrows=P0_P4_ROWS,
        low_memory=False,
    ).reset_index(names="row_index")
    if len(frame) != P0_P4_ROWS:
        raise AssertionError(f"Expected exactly {P0_P4_ROWS:,} P0-P4 rows.")

    definitions = json.loads(period_definitions.read_text(encoding="utf-8"))
    allowed = [item for item in definitions if item["period"] in EXPECTED_PERIOD_ROWS]

    def period_for_block(block: int) -> str:
        matches = [
            str(item["period"])
            for item in allowed
            if int(item["block_min"]) <= block <= int(item["block_max"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Block {block} is outside the frozen P0-P4 definitions.")
        return matches[0]

    frame["period"] = [period_for_block(int(value)) for value in frame["createdBlockNumber"]]
    if frame.groupby("period").size().to_dict() != EXPECTED_PERIOD_ROWS:
        raise AssertionError("Frozen Phase 7 P0-P4 memberships changed.")
    frame["tokens"] = frame["opcode"].map(canonical_opcode_tokens)
    return frame.sort_values(["createdBlockNumber", "row_index"], kind="mergesort").reset_index(drop=True)


def prepare_fold_corpus(
    frame: pd.DataFrame,
    *,
    fold: str,
    historical_periods: tuple[str, ...],
    chunk_length: int,
) -> PreparedCorpus:
    """Fit a vocabulary and construct chunks from historical rows only."""
    historical = frame[frame["period"].isin(historical_periods)]
    if historical.empty:
        raise ValueError(f"Fold {fold} has no historical contracts.")
    vocabulary = build_vocabulary(historical["tokens"])
    chunks: list[list[int]] = []
    chunk_counts: list[int] = []
    non_padding_tokens = 0
    for tokens in historical["tokens"]:
        contract_chunks = fixed_size_chunks(tokens, chunk_length)
        chunk_counts.append(len(contract_chunks))
        for chunk in contract_chunks:
            encoded = vocabulary.encode(chunk)
            non_padding_tokens += len(encoded)
            chunks.append(encoded + [vocabulary.pad_id] * (chunk_length - len(encoded)))
    chunk_array = np.asarray(chunks, dtype=np.uint16)
    if chunk_array.shape != (len(chunks), chunk_length):
        raise AssertionError("Chunk matrix shape is invalid.")
    return PreparedCorpus(
        fold=fold,
        historical_periods=historical_periods,
        row_indices=tuple(int(value) for value in historical["row_index"]),
        vocabulary=vocabulary,
        vocabulary_hash=vocabulary_sha256(vocabulary),
        chunks=chunk_array,
        contract_count=int(len(historical)),
        non_padding_tokens=int(non_padding_tokens),
        padding_tokens=int(chunk_array.size - non_padding_tokens),
        mean_chunks_per_contract=float(np.mean(chunk_counts)),
        max_chunks_per_contract=int(max(chunk_counts)),
    )


def seed_everything(seed: int) -> dict[str, Any]:
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": "warn_only",
        "bit_exact_gpu_retraining_claimed": False,
    }


def apply_mlm_mask(
    token_ids: torch.Tensor,
    *,
    vocabulary_size: int,
    mask_id: int,
    protected_token_count: int,
    probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    """Apply BERT-style 80/10/10 masking using one explicit generator."""
    if not math.isclose(probability, 0.15):
        raise ValueError("Phase 8R5B masking probability must remain 0.15.")
    if vocabulary_size <= protected_token_count:
        raise ValueError("The historical vocabulary has no ordinary opcode tokens.")
    eligible = token_ids.ge(protected_token_count)
    selected = torch.rand(token_ids.shape, device=token_ids.device, generator=generator) < probability
    selected &= eligible
    if not bool(selected.any()):
        first = eligible.nonzero(as_tuple=False)
        if len(first):
            selected[tuple(first[0])] = True
    inputs = token_ids.clone()
    outcomes = torch.rand(token_ids.shape, device=token_ids.device, generator=generator)
    mask_replaced = selected & outcomes.lt(0.80)
    random_replaced = selected & outcomes.ge(0.80) & outcomes.lt(0.90)
    unchanged = selected & outcomes.ge(0.90)
    inputs[mask_replaced] = mask_id
    random_values = torch.randint(
        protected_token_count,
        vocabulary_size,
        token_ids.shape,
        device=token_ids.device,
        generator=generator,
    )
    inputs[random_replaced] = random_values[random_replaced]
    counts = {
        "selected": int(selected.sum().item()),
        "mask_replaced": int(mask_replaced.sum().item()),
        "random_replaced": int(random_replaced.sum().item()),
        "unchanged": int(unchanged.sum().item()),
    }
    return inputs, token_ids, selected, counts


def encoder_hidden_states(
    model: OpcodeMaskedLanguageModel,
    token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Expose final-layer token states without adding a classifier."""
    embedded = model.position(model.embedding(token_ids))
    return model.encoder(embedded, src_key_padding_mask=padding_mask)


def masked_mean_pool(hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).unsqueeze(-1).to(dtype=hidden_states.dtype)
    denominator = valid.sum(dim=1).clamp_min(1.0)
    return (hidden_states * valid).sum(dim=1) / denominator


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gpu_utilization_percent() -> float | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip().splitlines()[0]
        return float(output)
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def train_pretraining_run(
    corpus: PreparedCorpus,
    *,
    seed: int,
    config: dict[str, Any],
    checkpoint_path: Path,
    preprocessing_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Train one fresh fold/seed model and save its epoch-10 checkpoint."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is prohibited.")
    device = torch.device(config["execution"]["device"])
    if torch.cuda.get_device_name(device) != config["execution"]["expected_gpu"]:
        raise RuntimeError("The configured RTX 2050 CUDA device is unavailable.")
    seed_record = seed_everything(seed)
    model_config = config["model"]
    model = OpcodeMaskedLanguageModel(
        len(corpus.vocabulary.token_to_id),
        maximum_length=int(config["sequence"]["length"]),
        embedding_dim=int(model_config["vocabulary_embedding_dim"]),
        encoder_layers=int(model_config["encoder_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        feed_forward_dim=int(model_config["feed_forward_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    initial_state_hash = model_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    use_amp = bool(config["execution"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batch_size = int(config["execution"]["physical_batch_size"])
    epochs = int(config["optimization"]["epochs"])
    epoch_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    max_allocated = 0
    max_reserved = 0
    utilization: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        order_generator = torch.Generator(device="cpu").manual_seed(seed * 100_000 + epoch)
        order = torch.randperm(len(corpus.chunks), generator=order_generator).numpy()
        mask_generator = torch.Generator(device=device).manual_seed(seed * 1_000_000 + epoch)
        loss_sum = 0.0
        correct = 0
        masked_count = 0
        tokens_processed = 0
        chunks_processed = 0
        replacement_counts = {"mask_replaced": 0, "random_replaced": 0, "unchanged": 0}
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            targets = torch.as_tensor(
                corpus.chunks[batch_indices].astype(np.int64, copy=False),
                device=device,
                dtype=torch.long,
            )
            padding_mask = targets.eq(corpus.vocabulary.pad_id)
            inputs, targets, selected, mask_counts = apply_mlm_mask(
                targets,
                vocabulary_size=len(corpus.vocabulary.token_to_id),
                mask_id=corpus.vocabulary.mask_id,
                protected_token_count=len(SPECIAL_TOKENS),
                probability=float(config["masking"]["probability"]),
                generator=mask_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(inputs, padding_mask)
                loss = masked_opcode_loss(logits, targets, selected)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite MLM loss in {corpus.fold}, seed {seed}.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            selected_count = int(selected.sum().item())
            with torch.no_grad():
                correct += int(logits[selected].argmax(dim=-1).eq(targets[selected]).sum().item())
            loss_sum += float(loss.detach().item()) * selected_count
            masked_count += selected_count
            tokens_processed += int((~padding_mask).sum().item())
            chunks_processed += int(len(batch_indices))
            for key in replacement_counts:
                replacement_counts[key] += mask_counts[key]

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        max_allocated = max(max_allocated, torch.cuda.max_memory_allocated(device))
        max_reserved = max(max_reserved, torch.cuda.max_memory_reserved(device))
        sampled_utilization = gpu_utilization_percent()
        if sampled_utilization is not None:
            utilization.append(sampled_utilization)
        average_loss = loss_sum / masked_count
        epoch_rows.append({
            "fold": corpus.fold,
            "seed": seed,
            "epoch": epoch,
            "mlm_training_loss": average_loss,
            "perplexity": math.exp(min(average_loss, 20.0)),
            "masked_token_top1_accuracy": correct / masked_count,
            "masked_token_count": masked_count,
            "mask_replaced_count": replacement_counts["mask_replaced"],
            "random_replaced_count": replacement_counts["random_replaced"],
            "unchanged_selected_count": replacement_counts["unchanged"],
            "tokens_processed": tokens_processed,
            "chunks_processed": chunks_processed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
            "examples_per_second": chunks_processed / elapsed,
            "tokens_per_second": tokens_processed / elapsed,
            "peak_allocated_vram_mib": max_allocated / (1024 ** 2),
            "peak_reserved_vram_mib": max_reserved / (1024 ** 2),
            "gpu_utilization_sample_percent": sampled_utilization,
            "validation_mlm_loss": None,
            "validation_masked_token_accuracy": None,
        })

    run_seconds = time.perf_counter() - run_start
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": {
            "schema_version": "phase8r5b_opcode_mlm_v1",
            "architecture": model_config,
            "vocabulary": corpus.vocabulary.token_to_id,
            "vocabulary_hash": corpus.vocabulary_hash,
            "fold": corpus.fold,
            "historical_periods": list(corpus.historical_periods),
            "historical_row_indices": list(corpus.row_indices),
            "seed": seed,
            "epoch": epochs,
            "optimizer": config["optimization"],
            "sequence_policy": config["sequence"],
            "masking_policy": config["masking"],
            "special_token_mapping": {
                token: corpus.vocabulary.token_to_id[token] for token in SPECIAL_TOKENS
            },
            "preprocessing_hashes": preprocessing_hashes,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "amp": use_amp,
            "embedding_readiness": config["embedding_readiness"],
            "gpu_bit_exact_retraining_claimed": False,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    run_record = {
        "run_id": f"{corpus.fold}_seed_{seed}",
        "fold": corpus.fold,
        "historical_periods": "+".join(corpus.historical_periods),
        "seed": seed,
        "status": "complete",
        "fresh_initialization": True,
        "optimizer_state_reused": False,
        "initial_model_state_hash": initial_state_hash,
        "epochs_completed": epochs,
        "final_training_loss": epoch_rows[-1]["mlm_training_loss"],
        "final_masked_token_accuracy": epoch_rows[-1]["masked_token_top1_accuracy"],
        "runtime_seconds": run_seconds,
        "average_gpu_utilization_percent": float(np.mean(utilization)) if utilization else None,
        "peak_allocated_vram_mib": max_allocated / (1024 ** 2),
        "peak_reserved_vram_mib": max_reserved / (1024 ** 2),
        "checkpoint_path": str(checkpoint_path),
        **seed_record,
    }
    resource_record = {
        "run_id": run_record["run_id"],
        "fold": corpus.fold,
        "seed": seed,
        "runtime_seconds": run_seconds,
        "average_gpu_utilization_percent": run_record["average_gpu_utilization_percent"],
        "peak_allocated_vram_mib": run_record["peak_allocated_vram_mib"],
        "peak_reserved_vram_mib": run_record["peak_reserved_vram_mib"],
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
    }
    del model, optimizer, scaler, checkpoint
    torch.cuda.empty_cache()
    return epoch_rows, run_record, resource_record
