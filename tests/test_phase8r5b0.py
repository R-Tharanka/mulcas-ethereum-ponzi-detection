from __future__ import annotations

import json

import pandas as pd
import torch

from src.extension.phase8r5b0 import (
    cuda_ready,
    deterministic_benchmark_chunks,
    feasibility_rating,
    is_cuda_oom,
    load_config,
    load_historical_sequences,
    verify_frozen_inputs,
)
from src.utils.paths import REPO_ROOT


def test_all_frozen_phase8r5a_anchors_verify() -> None:
    observed = verify_frozen_inputs(load_config())
    assert len(observed) >= 143


def test_configuration_freezes_architecture_and_scope() -> None:
    config = load_config()
    assert config["model"]["embedding_dim"] == 128
    assert config["model"]["encoder_layers"] == 2
    assert config["model"]["attention_heads"] == 4
    assert config["model"]["feed_forward_dim"] == 256
    assert config["model"]["dropout"] == 0.1
    assert config["benchmark"]["sequence_length"] == 1024
    assert config["benchmark"]["batch_sizes"] == [1, 2, 4, 8]
    assert config["execution"]["full_pretraining"] is False
    assert config["execution"]["supervised_training"] is False


def test_historical_loader_excludes_p5_and_labels() -> None:
    frame = load_historical_sequences(load_config())
    assert len(frame) == 5989
    assert "label" not in frame.columns
    assert frame["row_index"].max() == 5988


def test_benchmark_subset_is_deterministic() -> None:
    frame = pd.DataFrame({"tokens": [("ADD", "STOP"), ("PUSH1", "RETURN")]})
    first, first_vocabulary = deterministic_benchmark_chunks(frame, chunk_length=4, required_chunks=2)
    second, second_vocabulary = deterministic_benchmark_chunks(frame, chunk_length=4, required_chunks=2)
    assert first_vocabulary == second_vocabulary
    assert (first == second).all()


def test_cuda_gate_and_oom_detection_are_explicit() -> None:
    ready, reason = cuda_ready({"torch_cuda_available": False, "torch_cuda_device_name": None}, "NVIDIA GeForce RTX 2050")
    assert ready is False and "CPU-only" in reason
    assert is_cuda_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert is_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not is_cuda_oom(RuntimeError("different failure"))


def test_feasibility_rating_boundaries() -> None:
    assert feasibility_rating(2.0) == "GREEN"
    assert feasibility_rating(2.01) == "YELLOW"
    assert feasibility_rating(6.0) == "YELLOW"
    assert feasibility_rating(6.01) == "RED"


def test_generated_outputs_record_successful_bounded_gpu_execution() -> None:
    output = REPO_ROOT / "results/extension/phase8r5b0"
    results = json.loads((output / "phase8r5b0_results.json").read_text(encoding="utf-8"))
    assert results["status"] == "complete"
    assert results["cuda_environment_verified"] is True
    assert results["environment"]["torch_cuda_device_name"] == "NVIDIA GeForce RTX 2050"
    assert results["environment"]["torch_version"] == "2.14.1+cu130"
    assert results["gpu_benchmark_executed"] is True
    assert results["benchmark_data_loaded"] is True
    assert results["p5_loaded"] is False and results["labels_loaded"] is False
    assert results["full_pretraining_executed"] is False
    assert results["supervised_classifier_trained"] is False
    assert results["feasibility_rating"] in {"GREEN", "YELLOW", "RED"}
    assert results["phase8r5b_approved"] is False
    benchmark = pd.read_csv(output / "gpu_memory_benchmark.csv")
    assert not benchmark.empty
    assert set(benchmark["precision_mode"]) == {"fp32", "amp"}
    assert benchmark.loc[benchmark["status"].eq("success"), "loss_finite"].all()
    assert not pd.read_csv(output / "runtime_estimates.csv").empty


def test_source_has_no_cpu_fallback_or_full_pretraining_loop() -> None:
    source = (REPO_ROOT / "src/extension/phase8r5b0.py").read_text(encoding="utf-8")
    assert 'torch.device("cpu")' not in source
    assert "range(epochs)" not in source
    loader = source[source.index("def load_historical_sequences"):source.index("def deterministic_benchmark_chunks")]
    assert '"label"' not in loader
    assert "nrows=P0_P4_ROWS" in loader
