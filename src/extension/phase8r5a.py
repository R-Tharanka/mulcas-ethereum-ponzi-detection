"""Phase 8R5A opcode-sequence feasibility and compute audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import yaml

from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.extension.sequence_data import (
    SPECIAL_TOKENS,
    build_vocabulary,
    canonical_opcode_tokens,
    fixed_size_chunks,
    padded_ids,
)
from src.features.operands import decode_bytecode
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/extension/phase8r5a.yaml"
PERIODS = tuple(f"P{index}" for index in range(6))
EXPECTED_PERIOD_ROWS = {"P0": 104, "P1": 305, "P2": 5141, "P3": 289, "P4": 150, "P5": 509}
QUANTILES = (0.75, 0.90, 0.95, 0.99)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset", "period_definitions"):
        config["inputs"][key] = _resolve(config["inputs"][key])
    config["frozen_manifests"] = [_resolve(item) for item in config["frozen_manifests"]]
    config["output_dir"] = _resolve(config["output_dir"])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["tokenization"]["source_field"] != "opcode":
        raise ValueError("The deployed-runtime opcode field is the required sequence source.")
    if config["tokenization"]["raw_operands_allowed"]:
        raise ValueError("Raw operands are prohibited in Phase 8R5A.")
    if tuple(config["tokenization"]["special_tokens"]) != SPECIAL_TOKENS:
        raise ValueError("The fixed special-token policy changed.")
    execution = config["execution"]
    if any(execution[key] for key in ("supervised_classifier_training", "full_self_supervised_pretraining", "p5_model_evaluation")):
        raise ValueError("Phase 8R5A is a feasibility audit only.")
    if config["audit"]["p5_usage"] != "descriptive_only":
        raise ValueError("P5 may only be used for descriptive statistics.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in config["frozen_manifests"]:
        observed.update(verify_frozen_manifest(manifest))
    return observed


def _period_for_block(block: int, definitions: list[dict[str, Any]]) -> str:
    matches = [item["period"] for item in definitions if int(item["block_min"]) <= block <= int(item["block_max"])]
    if len(matches) != 1:
        raise ValueError(f"Block {block} does not have exactly one frozen period.")
    return str(matches[0])


def load_sequence_corpus(config: dict[str, Any]) -> pd.DataFrame:
    """Load label-free runtime sequences and frozen descriptive period membership."""
    frame = pd.read_csv(
        config["inputs"]["dataset"],
        usecols=["address", "createdBlockNumber", "opcode", "contractCode"],
        low_memory=False,
    ).reset_index(names="row_index")
    if len(frame) != 6498:
        raise AssertionError("Expected all 6,498 released contracts.")
    definitions = json.loads(config["inputs"]["period_definitions"].read_text(encoding="utf-8"))
    frame["period"] = [_period_for_block(int(block), definitions) for block in frame["createdBlockNumber"]]
    if frame.groupby("period").size().to_dict() != EXPECTED_PERIOD_ROWS:
        raise AssertionError("Frozen Phase 7 period memberships changed.")
    frame["tokens"] = frame["opcode"].map(canonical_opcode_tokens)
    frame["opcode_token_count"] = frame["tokens"].map(len).astype(int)
    frame["unique_opcode_count"] = frame["tokens"].map(lambda values: len(set(values))).astype(int)
    frame["runtime_byte_length"] = frame["contractCode"].map(
        lambda value: len(decoded) if (decoded := decode_bytecode(value)) is not None else 0
    ).astype(int)
    denominator = frame["opcode_token_count"].replace(0, np.nan)
    frame["sequence_byte_token_ratio"] = (frame["runtime_byte_length"] / denominator).fillna(0.0)
    return frame.sort_values(["createdBlockNumber", "row_index"], kind="mergesort").reset_index(drop=True)


def _distribution(values: pd.Series) -> dict[str, float | int]:
    numeric = values.astype(float)
    result: dict[str, float | int] = {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "median": float(numeric.median()),
        "mean": float(numeric.mean()),
        "max": float(numeric.max()),
    }
    for quantile in QUANTILES:
        result[f"p{int(quantile * 100)}"] = float(numeric.quantile(quantile))
    return result


def sequence_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ("opcode_token_count", "unique_opcode_count", "sequence_byte_token_ratio")
    global_rows = [{"scope": "ALL", "statistic": name, **_distribution(frame[name])} for name in columns]
    period_rows = [
        {"period": period, "statistic": name, **_distribution(group[name]), "design_use": period != "P5"}
        for period, group in frame.groupby("period", sort=True)
        for name in columns
    ]
    return pd.DataFrame(global_rows), pd.DataFrame(period_rows)


def truncation_coverage(frame: pd.DataFrame, lengths: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [
        ("ALL", frame, False),
        ("P0-P4_DESIGN", frame[~frame["period"].eq("P5")], True),
        *((period, frame[frame["period"].eq(period)], period != "P5") for period in PERIODS),
    ]
    for scope, group, design_use in scopes:
        counts = group["opcode_token_count"].to_numpy(dtype=np.int64)
        total_tokens = int(counts.sum())
        for maximum in lengths:
            retained = int(np.minimum(counts, maximum).sum())
            exceeds = int((counts > maximum).sum())
            rows.append({
                "scope": scope,
                "maximum_length": maximum,
                "contract_count": int(len(group)),
                "contracts_exceeding": exceeds,
                "contracts_exceeding_percent": 100.0 * exceeds / len(group),
                "total_tokens": total_tokens,
                "retained_tokens_prefix_or_prefix_suffix": retained,
                "retained_token_fraction": retained / total_tokens if total_tokens else 1.0,
                "fixed_chunking_retained_fraction": 1.0,
                "design_use": design_use,
            })
    return pd.DataFrame(rows)


def temporal_vocabulary_audit(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in config["temporal_vocabulary"]:
        historical = frame[frame["period"].isin(fold["training_periods"])]
        evaluation = frame[frame["period"].eq(fold["evaluation_period"])]
        vocabulary = build_vocabulary(historical["tokens"])
        unknown = {
            token for sequence in evaluation["tokens"] for token in sequence
            if token not in vocabulary.token_to_id
        }
        oov_count = sum(token in unknown for sequence in evaluation["tokens"] for token in sequence)
        evaluation_tokens = int(evaluation["opcode_token_count"].sum())
        rows.append({
            "fold": fold["fold"],
            "training_periods": "+".join(fold["training_periods"]),
            "evaluation_period": fold["evaluation_period"],
            "historical_rows": int(len(historical)),
            "evaluation_rows": int(len(evaluation)),
            "historical_vocabulary_size_including_special_tokens": len(vocabulary.token_to_id),
            "historical_opcode_vocabulary_size": len(vocabulary.token_to_id) - len(SPECIAL_TOKENS),
            "evaluation_token_count": evaluation_tokens,
            "evaluation_oov_token_count": int(oov_count),
            "evaluation_oov_token_rate": oov_count / evaluation_tokens if evaluation_tokens else 0.0,
            "evaluation_unique_oov_tokens": "|".join(sorted(unknown)),
            "evaluation_tokens_used_for_fit": False,
        })
    return pd.DataFrame(rows)


class TinyOpcodeEncoder(nn.Module):
    """Benchmark-only encoder matching the proposed Phase 8R5B dimensions."""

    def __init__(self, vocabulary_size: int, model_config: dict[str, Any]) -> None:
        super().__init__()
        embedding_dim = int(model_config["embedding_dim"])
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=int(model_config["attention_heads"]),
            dim_feedforward=int(model_config["feed_forward_dim"]),
            dropout=float(model_config["dropout"]),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(model_config["encoder_layers"]))
        self.output = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(self.embedding(token_ids), src_key_padding_mask=padding_mask))


def _gpu_environment() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    try:
        line = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()[0]
        name, memory, driver = [value.strip() for value in line.split(",", maxsplit=2)]
        return {"system_gpu_detected": True, "gpu_model": name, "gpu_vram_mib": int(memory), "gpu_driver": driver}
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return {"system_gpu_detected": False, "gpu_model": None, "gpu_vram_mib": None, "gpu_driver": None}


def runtime_environment() -> dict[str, Any]:
    gpu = _gpu_environment()
    try:
        import torch_geometric
        geometric_version: str | None = torch_geometric.__version__
    except ImportError:
        geometric_version = None
    cuda_available = bool(torch.cuda.is_available())
    if gpu["system_gpu_detected"] and not cuda_available and "+cpu" in torch.__version__:
        gpu_status = "gpu_present_but_installed_torch_is_cpu_only"
    elif not gpu["system_gpu_detected"]:
        gpu_status = "no_compatible_gpu_detected"
    else:
        gpu_status = "cuda_available" if cuda_available else "gpu_detected_but_cuda_unavailable"
    return {
        "os": platform.platform(),
        "cpu_model": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "system_ram_gib": psutil.virtual_memory().total / (1024 ** 3),
        "torch_version": torch.__version__,
        "torch_build_type": "cpu" if "+cpu" in torch.__version__ else "cuda_or_other",
        "torch_cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "torch_geometric_version": geometric_version,
        "execution_device": "cpu",
        "gpu_status": gpu_status,
        **gpu,
    }


def run_tiny_benchmark(frame: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = config["benchmark"]
    historical = frame[frame["period"].isin(benchmark["historical_periods"])]
    vocabulary = build_vocabulary(historical["tokens"])
    source_tokens = max(historical["tokens"], key=len)
    rows: list[dict[str, Any]] = []
    for length in benchmark["sequence_lengths"]:
        set_random_seed(int(config["random_seed"]))
        ids, padding = padded_ids(source_tokens, vocabulary, int(length))
        token_ids = torch.tensor([ids], dtype=torch.long)
        padding_mask = torch.tensor([padding], dtype=torch.bool)
        targets = token_ids.clone()
        positions = torch.arange(int(length)) % int(benchmark["mask_stride"]) == 0
        positions &= ~padding_mask[0]
        masked_ids = token_ids.clone()
        masked_ids[0, positions] = vocabulary.mask_id
        model = TinyOpcodeEncoder(len(vocabulary.token_to_id), config["future_model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["future_model"]["learning_rate"]))
        process = psutil.Process(os.getpid())
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(masked_ids, padding_mask)
        loss = nn.functional.cross_entropy(logits[0, positions], targets[0, positions])
        loss.backward()
        optimizer.step()
        elapsed = time.perf_counter() - started
        memory = process.memory_info()
        rows.append({
            "sequence_length": int(length),
            "batch_size": int(benchmark["batch_size"]),
            "batches": int(benchmark["fixed_batches"]),
            "device": "cpu",
            "elapsed_seconds": elapsed,
            "examples_per_second": int(benchmark["batch_size"]) / elapsed,
            "tokens_per_second": int(length) * int(benchmark["batch_size"]) / elapsed,
            "rss_delta_mib": max(0, memory.rss - rss_before) / (1024 ** 2),
            "process_peak_working_set_mib": getattr(memory, "peak_wset", memory.rss) / (1024 ** 2),
            "masked_tokens": int(positions.sum()),
            "loss": float(loss.detach()),
            "full_pretraining": False,
            "supervised_training": False,
        })
        del model, optimizer, logits, loss
    return rows


def runtime_estimates(frame: pd.DataFrame, benchmark_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    chunk_length = int(config["audit"]["recommended_chunk_length"])
    speed = next(row["examples_per_second"] for row in benchmark_rows if row["sequence_length"] == chunk_length)
    epochs = int(config["future_model"]["planned_pretraining_epochs"])
    fold_rows: list[dict[str, Any]] = []
    for fold in config["temporal_vocabulary"]:
        historical = frame[frame["period"].isin(fold["training_periods"])]
        chunks = int(sum(max(1, math.ceil(count / chunk_length)) for count in historical["opcode_token_count"]))
        seconds_per_epoch = chunks / speed
        fold_rows.append({
            "prediction_fold": fold["fold"],
            "historical_periods": fold["training_periods"],
            "historical_contracts": int(len(historical)),
            "estimated_chunks_per_epoch": chunks,
            "estimated_seconds_per_epoch_cpu": seconds_per_epoch,
            "planned_epochs": epochs,
            "estimated_seconds_one_fold_cpu": seconds_per_epoch * epochs,
        })
    one_seed = sum(row["estimated_seconds_one_fold_cpu"] for row in fold_rows)
    three_seeds = one_seed * len(config["future_model"]["future_seeds"])
    threshold = float(config["execution"]["runtime_approval_threshold_seconds"])
    return {
        "basis": "linear estimate from one measured batch at the recommended 1024-token chunk length",
        "limitations": "excludes data-loader, checkpoint, validation, and repeated-epoch caching effects",
        "folds": fold_rows,
        "estimated_seconds_all_p2_p4_one_seed_cpu": one_seed,
        "estimated_seconds_all_p2_p4_three_seeds_cpu": three_seeds,
        "gpu_estimate": None,
        "gpu_estimate_reason": "No CUDA-capable PyTorch benchmark was available.",
        "exceeds_30_minutes_one_seed": one_seed > threshold,
        "exceeds_30_minutes_three_seeds": three_seeds > threshold,
        "recommendation": (
            "Before Phase 8R5B, install a compatible CUDA PyTorch build in a separately verified environment, benchmark again, and seek explicit approval."
            if one_seed > threshold or three_seeds > threshold
            else "The measured linear estimate is below 30 minutes; still rerun a representative pretraining epoch before Phase 8R5B."
        ),
    }


def build_report(results: dict[str, Any], lengths: pd.DataFrame, vocab: pd.DataFrame) -> str:
    token_stats = results["global_opcode_token_count"]
    benchmark = results["compute"]["benchmark"]
    estimate = results["compute"]["runtime_estimates"]
    lines = [
        "# Phase 8R5A Opcode-Sequence Feasibility",
        "",
        "## Scope",
        "",
        "This phase audited deployed-runtime opcode sequences and compute feasibility only. No supervised Ponzi classifier was trained, no full masked-opcode pretraining was launched, and P5 was used only in global descriptive statistics.",
        "",
        "## Frozen Research",
        "",
        f"All configured Phase 0-8R4B anchors verified ({results['frozen_artifact_count']} unique files). The raw dataset SHA-256 remained `{results['dataset_sha256']}`.",
        "",
        "## Sequence Audit",
        "",
        f"Across 6,498 contracts, opcode-token length ranged from {token_stats['min']:.0f} to {token_stats['max']:.0f}, with median {token_stats['median']:.1f}, mean {token_stats['mean']:.1f}, P90 {token_stats['p90']:.1f}, P95 {token_stats['p95']:.1f}, and P99 {token_stats['p99']:.1f}.",
        "",
        "The policy decision uses P0-P4 only; P5 and all-contract values are descriptive.",
        "",
        "| P0-P4 limit | Contracts exceeding | Percent | Tokens retained by truncation |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in lengths[lengths["scope"].eq("P0-P4_DESIGN")].itertuples(index=False):
        lines.append(f"| {row.maximum_length} | {row.contracts_exceeding} | {row.contracts_exceeding_percent:.2f}% | {row.retained_token_fraction:.4f} |")
    lines.extend(["", "| Period | Median tokens | Mean tokens | P95 | Max |", "| --- | ---: | ---: | ---: | ---: |"])
    period_stats = results["period_opcode_token_count"]
    for period in PERIODS:
        values = period_stats[period]
        suffix = " (descriptive only)" if period == "P5" else ""
        lines.append(f"| {period}{suffix} | {values['median']:.1f} | {values['mean']:.1f} | {values['p95']:.1f} | {values['max']:.0f} |")
    lines.extend(["", "## Temporal Vocabulary", "", "| Evaluation | Historical periods | Vocabulary | OOV tokens | OOV rate |", "| --- | --- | ---: | ---: | ---: |"])
    for row in vocab.itertuples(index=False):
        lines.append(f"| {row.evaluation_period} | {row.training_periods} | {row.historical_opcode_vocabulary_size} | {row.evaluation_oov_token_count} | {row.evaluation_oov_token_rate:.6f} |")
    lines.extend([
        "",
        "## Representation Recommendation",
        "",
        "Use deterministic 1,024-token chunking with future chunk-level pooling. It preserves all tokens, avoids the information loss of truncation, has lower quadratic attention cost than a single 2,048/4,096-token sequence, and remains label-independent. Prefix and prefix+suffix remain documented alternatives, not performance-selected policies.",
        "",
        "## Compute Preflight",
        "",
        f"PyTorch `{results['compute']['environment']['torch_version']}` reports CUDA unavailable. The OS sees `{results['compute']['environment']['gpu_model']}` with {results['compute']['environment']['gpu_vram_mib']} MiB, so the installed Torch build is CPU-only despite a GPU being present.",
        "",
        "| Length | Seconds/batch | Examples/s | Tokens/s | RSS delta MiB |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in benchmark:
        lines.append(f"| {row['sequence_length']} | {row['elapsed_seconds']:.4f} | {row['examples_per_second']:.4f} | {row['tokens_per_second']:.1f} | {row['rss_delta_mib']:.1f} |")
    lines.extend([
        "",
        f"Estimated P2-P4 pretraining time for one seed and {results['model_plan']['planned_pretraining_epochs']} epochs/fold: {estimate['estimated_seconds_all_p2_p4_one_seed_cpu'] / 60:.1f} minutes. Three seeds: {estimate['estimated_seconds_all_p2_p4_three_seeds_cpu'] / 3600:.1f} hours.",
        "",
        f"Recommendation: {estimate['recommendation']}",
        "",
        "## Frozen Future Design",
        "",
        "The proposed encoder uses 128-dimensional opcode embeddings, two Transformer encoder layers, four attention heads, a 256-dimensional feed-forward layer, dropout 0.1, and masked-opcode prediction. Each P2-P4 fold must build its vocabulary and pretraining corpus from completed historical periods only.",
        "",
        "H8R5 is not evaluated in Phase 8R5A.",
        "",
        "## Blockers Before Phase 8R5B",
        "",
        "- The installed PyTorch build cannot use the available GPU.",
        "- The CPU estimate exceeds the 30-minute automatic-launch limit by a wide margin.",
        "- A CUDA-compatible environment needs a fresh representative benchmark before choosing batch size or launching pretraining.",
        "- Phase 8R5B requires explicit approval after that benchmark and must keep separate historical pretraining corpora for P2, P3, and P4.",
    ])
    return "\n".join(lines) + "\n"


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    frozen = verify_frozen_inputs(config)
    frame = load_sequence_corpus(config)
    global_stats, period_stats = sequence_statistics(frame)
    coverage = truncation_coverage(frame, [int(value) for value in config["audit"]["candidate_max_lengths"]])
    vocabulary = temporal_vocabulary_audit(frame, config)
    environment = runtime_environment()
    benchmark = run_tiny_benchmark(frame, config)
    estimates = runtime_estimates(frame, benchmark, config)
    model_plan = {
        **config["future_model"],
        "vocabulary_policy": "fit separately on each fold's historical periods only",
        "sequence_policy": "deterministic 1024-token chunks with future chunk pooling",
        "p5_allowed": False,
        "architecture_search": False,
    }
    results = {
        "phase": "8R5A",
        "hypothesis": config["hypothesis"],
        "hypothesis_evaluated": False,
        "dataset_sha256": sha256_file(config["inputs"]["dataset"]),
        "config_sha256": config["config_sha256"],
        "frozen_artifacts_verified": True,
        "frozen_artifact_count": len(frozen),
        "contracts_audited": len(frame),
        "token_source": "deployed-runtime opcode field, canonical names with hexadecimal operands removed",
        "special_tokens": list(SPECIAL_TOKENS),
        "global_opcode_token_count": _distribution(frame["opcode_token_count"]),
        "period_opcode_token_count": {
            period: _distribution(frame.loc[frame["period"].eq(period), "opcode_token_count"])
            for period in PERIODS
        },
        "recommended_sequence_policy": "fixed 1024-token chunking followed by future chunk pooling",
        "recommendation_basis": ["P0-P4 coverage only", "100% token retention", "bounded quadratic attention cost", "simplicity", "determinism"],
        "p5_usage": "descriptive_only",
        "p5_model_evaluation": False,
        "p5_in_model_design": False,
        "supervised_classifier_trained": False,
        "full_self_supervised_pretraining_run": False,
        "compute": {"environment": environment, "benchmark": benchmark, "runtime_estimates": estimates},
        "model_plan": model_plan,
        "actual_phase_runtime_seconds": time.perf_counter() - started,
    }
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    global_stats.to_csv(output / "sequence_length_statistics.csv", index=False)
    period_stats.to_csv(output / "period_sequence_statistics.csv", index=False)
    vocabulary.to_csv(output / "temporal_vocabulary_audit.csv", index=False)
    coverage.to_csv(output / "truncation_coverage.csv", index=False)
    (output / "runtime_preflight.json").write_text(json.dumps(results["compute"], indent=2), encoding="utf-8")
    (output / "sequence_model_plan.json").write_text(json.dumps(model_plan, indent=2), encoding="utf-8")
    (output / "phase8r5a_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "phase8r5a_report.md").write_text(build_report(results, coverage, vocabulary), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    results = run(args.config)
    print(json.dumps({
        "contracts_audited": results["contracts_audited"],
        "gpu_status": results["compute"]["environment"]["gpu_status"],
        "recommended_sequence_policy": results["recommended_sequence_policy"],
        "full_pretraining_run": results["full_self_supervised_pretraining_run"],
    }, indent=2))


if __name__ == "__main__":
    main()
