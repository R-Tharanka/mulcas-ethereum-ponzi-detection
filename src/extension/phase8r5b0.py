"""CUDA validation and bounded GPU benchmark for the frozen opcode Transformer."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.extension.sequence_data import build_vocabulary, canonical_opcode_tokens, fixed_size_chunks
from src.extension.sequence_model import OpcodeMaskedLanguageModel, masked_opcode_loss
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/extension/phase8r5b0.yaml"
P0_P4_ROWS = 5989
BENCHMARK_COLUMNS = (
    "precision_mode", "batch_size", "status", "peak_allocated_mib",
    "peak_reserved_mib", "vram_headroom_fraction", "examples_per_second",
    "tokens_per_second", "mean_forward_seconds", "mean_backward_seconds",
    "mean_total_batch_seconds", "loss_finite", "mean_loss", "error",
)
RUNTIME_COLUMNS = (
    "precision_mode", "batch_size", "prediction_fold", "chunk_count",
    "epochs", "estimated_seconds", "estimated_minutes", "estimated_hours",
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in config["inputs"]:
        config["inputs"][key] = _resolve(config["inputs"][key])
    config["frozen_manifests"] = [_resolve(item) for item in config["frozen_manifests"]]
    config["output_dir"] = _resolve(config["output_dir"])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    expected_model = {
        "vocabulary_size": 145, "embedding_dim": 128,
        "positional_encoding": "sinusoidal", "encoder_layers": 2,
        "attention_heads": 4, "feed_forward_dim": 256, "dropout": 0.1,
        "objective": "masked_opcode_prediction", "classification_head": False,
    }
    if config["model"] != expected_model:
        raise ValueError("The frozen Phase 8R5 architecture changed.")
    benchmark = config["benchmark"]
    if benchmark["sequence_length"] != 1024 or benchmark["batch_sizes"] != [1, 2, 4, 8]:
        raise ValueError("The frozen sequence length or batch-size audit changed.")
    if benchmark["p5_allowed"] or benchmark["labels_allowed"]:
        raise ValueError("P5 and labels are prohibited in Phase 8R5B0.")
    if any(config["execution"][key] for key in ("full_pretraining", "supervised_training", "p5_evaluation")):
        raise ValueError("Phase 8R5B0 is a bounded systems benchmark only.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in config["frozen_manifests"]:
        observed.update(verify_frozen_manifest(manifest))
    return observed


def _nvidia_environment() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.run(query, capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()[0]
        name, memory, driver = (part.strip() for part in line.split(",", maxsplit=2))
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=True).stdout
        cuda_visibility = next((part.split()[0] for part in header.split("CUDA UMD Version:")[1:]), None)
        return {
            "nvidia_smi_available": True,
            "gpu_model": name,
            "gpu_vram_mib": int(memory),
            "nvidia_driver_version": driver,
            "cuda_driver_visibility": cuda_visibility,
        }
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return {
            "nvidia_smi_available": False, "gpu_model": None,
            "gpu_vram_mib": None, "nvidia_driver_version": None,
            "cuda_driver_visibility": None,
        }


def collect_environment() -> dict[str, Any]:
    try:
        import torch_geometric
        pyg_version: str | None = torch_geometric.__version__
    except ImportError:
        pyg_version = None
    try:
        import torchvision
        torchvision_version: str | None = torchvision.__version__
    except ImportError:
        torchvision_version = None
    try:
        import torchaudio
        torchaudio_version: str | None = torchaudio.__version__
    except ImportError:
        torchaudio_version = None
    cuda_available = bool(torch.cuda.is_available())
    return {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "system_ram_gib": psutil.virtual_memory().total / (1024 ** 3),
        "torch_version": torch.__version__,
        "torch_build_type": "cpu" if "+cpu" in torch.__version__ else "cuda_or_other",
        "torch_cuda_available": cuda_available,
        "torch_cuda_runtime_version": torch.version.cuda,
        "torch_cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "torchvision_version": torchvision_version,
        "torchaudio_version": torchaudio_version,
        "torch_geometric_version": pyg_version,
        **_nvidia_environment(),
    }


def cuda_ready(environment: dict[str, Any], expected_gpu: str) -> tuple[bool, str]:
    if not environment["torch_cuda_available"]:
        return False, "Installed PyTorch is CPU-only or cannot initialize CUDA."
    if environment["torch_cuda_device_name"] != expected_gpu:
        return False, f"CUDA device mismatch: {environment['torch_cuda_device_name']!r}."
    return True, "CUDA device verified."


def load_historical_sequences(config: dict[str, Any]) -> pd.DataFrame:
    """Load only the exact frozen P0-P4 prefix, with no labels or P5 rows."""
    frame = pd.read_csv(
        config["inputs"]["dataset"],
        usecols=["createdBlockNumber", "opcode"],
        nrows=P0_P4_ROWS,
        low_memory=False,
    ).reset_index(names="row_index")
    if len(frame) != P0_P4_ROWS:
        raise AssertionError("Expected exactly the frozen 5,989 P0-P4 rows.")
    definitions = json.loads(config["inputs"]["period_definitions"].read_text(encoding="utf-8"))
    p4_max = next(int(item["block_max"]) for item in definitions if item["period"] == "P4")
    if int(frame["createdBlockNumber"].max()) > p4_max:
        raise AssertionError("A P5 row entered the benchmark corpus.")
    frame["tokens"] = frame["opcode"].map(canonical_opcode_tokens)
    return frame


def deterministic_benchmark_chunks(
    frame: pd.DataFrame, *, chunk_length: int, required_chunks: int
) -> tuple[np.ndarray, int]:
    vocabulary = build_vocabulary(frame["tokens"])
    chunks: list[list[int]] = []
    for sequence in frame["tokens"]:
        for chunk in fixed_size_chunks(sequence, chunk_length):
            ids = vocabulary.encode(chunk)[:chunk_length]
            chunks.append(ids + [vocabulary.pad_id] * (chunk_length - len(ids)))
            if len(chunks) == required_chunks:
                return np.asarray(chunks, dtype=np.int64), len(vocabulary.token_to_id)
    raise ValueError("Historical corpus did not provide enough benchmark chunks.")


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _benchmark_one(
    token_array: np.ndarray,
    vocabulary_size: int,
    batch_size: int,
    precision_mode: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    sequence_length = int(config["benchmark"]["sequence_length"])
    warmup = int(config["benchmark"]["warmup_batches"])
    measured = int(config["benchmark"]["measured_batches"])
    seed = int(config["random_seed"])
    set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    model = OpcodeMaskedLanguageModel(vocabulary_size, maximum_length=sequence_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    use_amp = precision_mode == "amp"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    forward_times: list[float] = []
    backward_times: list[float] = []
    total_times: list[float] = []
    losses: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup + measured):
        start = (step * batch_size) % len(token_array)
        indices = np.arange(start, start + batch_size) % len(token_array)
        targets = torch.as_tensor(token_array[indices], device=device, dtype=torch.long)
        inputs = targets.clone()
        masked_positions = torch.zeros_like(inputs, dtype=torch.bool)
        masked_positions[:, ::7] = True
        inputs[masked_positions] = 1
        optimizer.zero_grad(set_to_none=True)
        total_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(inputs)
            loss = masked_opcode_loss(logits, targets, masked_positions)
        forward_end.record()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        backward_end.record()
        torch.cuda.synchronize(device)
        if step >= warmup:
            forward = total_start.elapsed_time(forward_end) / 1000.0
            backward = forward_end.elapsed_time(backward_end) / 1000.0
            forward_times.append(forward)
            backward_times.append(backward)
            total_times.append(forward + backward)
            losses.append(float(loss.detach()))
    total = float(np.mean(total_times))
    properties = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        "precision_mode": precision_mode,
        "batch_size": batch_size,
        "status": "success",
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
        "vram_headroom_fraction": max(0.0, 1.0 - reserved * 1024 ** 2 / properties.total_memory),
        "examples_per_second": batch_size / total,
        "tokens_per_second": batch_size * sequence_length / total,
        "mean_forward_seconds": float(np.mean(forward_times)),
        "mean_backward_seconds": float(np.mean(backward_times)),
        "mean_total_batch_seconds": total,
        "loss_finite": bool(np.isfinite(losses).all()),
        "mean_loss": float(np.mean(losses)),
        "error": None,
    }


def run_gpu_benchmarks(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    maximum_chunks = max(config["benchmark"]["batch_sizes"]) * (
        int(config["benchmark"]["warmup_batches"]) + int(config["benchmark"]["measured_batches"])
    )
    chunks, vocabulary_size = deterministic_benchmark_chunks(
        frame, chunk_length=int(config["benchmark"]["sequence_length"]), required_chunks=maximum_chunks
    )
    if vocabulary_size != int(config["model"]["vocabulary_size"]):
        raise AssertionError("Historical opcode vocabulary no longer matches the frozen size.")
    rows: list[dict[str, Any]] = []
    for mode in config["benchmark"]["precision_modes"]:
        for batch_size in config["benchmark"]["batch_sizes"]:
            try:
                row = _benchmark_one(chunks, vocabulary_size, int(batch_size), mode, config)
            except RuntimeError as error:
                if not is_cuda_oom(error):
                    raise
                row = {column: None for column in BENCHMARK_COLUMNS}
                row.update({"precision_mode": mode, "batch_size": batch_size, "status": "oom", "loss_finite": False, "error": str(error)})
                rows.append(row)
                torch.cuda.empty_cache()
                break
            rows.append(row)
    return pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)


def select_execution_policy(benchmarks: pd.DataFrame) -> dict[str, Any] | None:
    successful = benchmarks[
        benchmarks["status"].eq("success")
        & benchmarks["loss_finite"].eq(True)
        & benchmarks["vram_headroom_fraction"].ge(0.10)
    ]
    if successful.empty:
        return None
    preferred = successful[successful["precision_mode"].eq("amp")]
    choices = preferred if not preferred.empty else successful
    selected = choices.sort_values(["batch_size", "tokens_per_second"], ascending=False).iloc[0]
    return selected.to_dict()


def build_runtime_estimates(
    benchmarks: pd.DataFrame, phase8r5a: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    successful = benchmarks[benchmarks["status"].eq("success")]
    chunks = {
        item["prediction_fold"]: int(item["estimated_chunks_per_epoch"])
        for item in phase8r5a["compute"]["runtime_estimates"]["folds"]
    }
    for benchmark in successful.itertuples(index=False):
        for fold, chunk_count in chunks.items():
            for epochs in config["runtime_estimation"]["epoch_counts"]:
                seconds = chunk_count / float(benchmark.examples_per_second) * int(epochs)
                rows.append({
                    "precision_mode": benchmark.precision_mode,
                    "batch_size": int(benchmark.batch_size),
                    "prediction_fold": fold,
                    "chunk_count": chunk_count,
                    "epochs": int(epochs),
                    "estimated_seconds": seconds,
                    "estimated_minutes": seconds / 60,
                    "estimated_hours": seconds / 3600,
                })
    return pd.DataFrame(rows, columns=RUNTIME_COLUMNS)


def feasibility_rating(one_seed_hours: float) -> str:
    if one_seed_hours <= 2:
        return "GREEN"
    if one_seed_hours <= 6:
        return "YELLOW"
    return "RED"


def _environment_snapshot(environment: dict[str, Any]) -> str:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    return json.dumps(environment, indent=2) + "\n\n[pip freeze]\n" + freeze


def _write_empty_tables(output: Path) -> None:
    pd.DataFrame(columns=BENCHMARK_COLUMNS).to_csv(output / "gpu_memory_benchmark.csv", index=False)
    pd.DataFrame(columns=BENCHMARK_COLUMNS).to_csv(output / "throughput_benchmark.csv", index=False)
    pd.DataFrame(columns=RUNTIME_COLUMNS).to_csv(output / "runtime_estimates.csv", index=False)


def _blocked_report(environment: dict[str, Any], frozen_count: int, reason: str) -> str:
    return f"""# Phase 8R5B0 CUDA Validation

## Outcome: BLOCKED

All {frozen_count} configured Phase 0-8R5A artifact anchors verified. CUDA GPU benchmarking did not run because the required CUDA-enabled PyTorch environment could not be established safely.

The machine reports an NVIDIA GeForce RTX 2050 with 4,096 MiB VRAM, NVIDIA driver `{environment['nvidia_driver_version']}`, and CUDA driver visibility `{environment['cuda_driver_visibility']}`. The project environment still contains PyTorch `{environment['torch_version']}`, for which `torch.cuda.is_available()` is `{environment['torch_cuda_available']}` and `torch.version.cuda` is `{environment['torch_cuda_runtime_version']}`.

The official `torch 2.14.1+cu130` wheel download was attempted from PyTorch's CUDA 13.0 index. The 1.99 GB transfer repeatedly lost connectivity and DNS resolution after approximately 1.13 GB. The installer was stopped before package replacement; `pip check` confirmed the original environment remains consistent.

Reason: {reason}

No P0-P4 benchmark corpus was loaded, no P5 row or label was loaded, no masked-opcode pretraining occurred, and no supervised classifier was trained. GPU throughput, VRAM, runtime estimates, batch policy, and GREEN/YELLOW/RED rating are therefore unavailable rather than estimated from CPU measurements.

## Required Next Step

Complete installation from `requirements-cuda.txt` on a stable connection, verify the RTX 2050 with CUDA, and rerun Phase 8R5B0. Full Phase 8R5B remains prohibited until a successful benchmark report receives explicit approval.
"""


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    frozen = verify_frozen_inputs(config)
    environment = collect_environment()
    ready, reason = cuda_ready(environment, config["cuda"]["expected_gpu"])
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "cuda_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (output / "environment_snapshot.txt").write_text(_environment_snapshot(environment), encoding="utf-8")
    if not ready:
        _write_empty_tables(output)
        results = {
            "phase": "8R5B0",
            "status": "blocked",
            "blocker": reason,
            "cuda_installation_attempt": config["cuda"]["installation_attempt_status"],
            "requested_cuda_torch": config["cuda"]["expected_torch"],
            "frozen_artifacts_verified": True,
            "frozen_artifact_count": len(frozen),
            "cuda_environment_verified": False,
            "gpu_benchmark_executed": False,
            "benchmark_data_loaded": False,
            "p5_loaded": False,
            "labels_loaded": False,
            "full_pretraining_executed": False,
            "supervised_classifier_trained": False,
            "successful_batch_sizes": [],
            "recommended_batch_policy": None,
            "runtime_estimates_available": False,
            "feasibility_rating": "BLOCKED",
            "phase8r5b_approved": False,
            "environment": environment,
        }
        (output / "phase8r5b0_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (output / "phase8r5b0_report.md").write_text(
            _blocked_report(environment, len(frozen), reason), encoding="utf-8"
        )
        return results

    frame = load_historical_sequences(config)
    benchmarks = run_gpu_benchmarks(frame, config)
    phase8r5a = json.loads(config["inputs"]["phase8r5a_results"].read_text(encoding="utf-8"))
    estimates = build_runtime_estimates(benchmarks, phase8r5a, config)
    policy = select_execution_policy(benchmarks)
    if policy is None:
        rating = "BLOCKED"
        one_seed_hours = None
    else:
        selected = estimates[
            estimates["precision_mode"].eq(policy["precision_mode"])
            & estimates["batch_size"].eq(policy["batch_size"])
            & estimates["epochs"].eq(10)
        ]
        one_seed_hours = float(selected["estimated_hours"].sum())
        rating = feasibility_rating(one_seed_hours)
    benchmarks.to_csv(output / "gpu_memory_benchmark.csv", index=False)
    benchmarks.to_csv(output / "throughput_benchmark.csv", index=False)
    estimates.to_csv(output / "runtime_estimates.csv", index=False)
    results = {
        "phase": "8R5B0", "status": "complete",
        "frozen_artifacts_verified": True, "frozen_artifact_count": len(frozen),
        "cuda_environment_verified": True, "gpu_benchmark_executed": True,
        "benchmark_data_loaded": True, "p5_loaded": False, "labels_loaded": False,
        "full_pretraining_executed": False, "supervised_classifier_trained": False,
        "successful_batch_sizes": benchmarks.loc[benchmarks["status"].eq("success"), "batch_size"].astype(int).tolist(),
        "recommended_batch_policy": policy,
        "one_seed_ten_epoch_hours": one_seed_hours,
        "three_seed_ten_epoch_hours": one_seed_hours * 3 if one_seed_hours is not None else None,
        "feasibility_rating": rating, "phase8r5b_approved": False,
        "environment": environment,
    }
    (output / "phase8r5b0_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "phase8r5b0_report.md").write_text(
        "# Phase 8R5B0 CUDA Validation\n\nGPU benchmark completed. See machine-readable outputs. Full Phase 8R5B remains approval-gated.\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
