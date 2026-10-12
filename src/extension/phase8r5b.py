"""Phase 8R5B temporally valid masked-opcode pretraining runner."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
import yaml

from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.extension.sequence_data import SPECIAL_TOKENS
from src.extension.sequence_pretraining import (
    PreparedCorpus,
    load_p0_p4_sequences,
    prepare_fold_corpus,
    train_pretraining_run,
)
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/extension/phase8r5b.yaml"
EXPECTED_SEEDS = [42, 123, 2026]
EXPECTED_FOLDS = {
    "P2": ("P0", "P1"),
    "P3": ("P0", "P1", "P2"),
    "P4": ("P0", "P1", "P2", "P3"),
}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in config["inputs"]:
        config["inputs"][key] = _resolve(config["inputs"][key])
    config["frozen_manifests"] = [_resolve(item) for item in config["frozen_manifests"]]
    config["checkpoint_dir"] = _resolve(config["checkpoint_dir"])
    config["output_dir"] = _resolve(config["output_dir"])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["random_seeds"] != EXPECTED_SEEDS:
        raise ValueError("Phase 8R5B requires exactly seeds 42, 123, and 2026.")
    observed_folds = {
        fold: tuple(values["historical_periods"]) for fold, values in config["folds"].items()
    }
    if observed_folds != EXPECTED_FOLDS:
        raise ValueError("The frozen P2-P4 historical folds changed.")
    if int(config["sequence"]["length"]) != 1024 or config["sequence"]["policy"] != "deterministic_fixed_size_chunking":
        raise ValueError("The frozen 1,024-token chunking policy changed.")
    if tuple(config["sequence"]["special_tokens"]) != SPECIAL_TOKENS:
        raise ValueError("The frozen special-token policy changed.")
    expected_model = (128, 2, 4, 256, 0.1)
    observed_model = (
        config["model"]["vocabulary_embedding_dim"], config["model"]["encoder_layers"],
        config["model"]["attention_heads"], config["model"]["feed_forward_dim"],
        config["model"]["dropout"],
    )
    if observed_model != expected_model or config["model"]["classification_head"]:
        raise ValueError("The frozen Transformer architecture changed.")
    masking = config["masking"]
    if [masking[key] for key in ("probability", "mask_replacement_probability", "random_replacement_probability", "unchanged_probability")] != [0.15, 0.80, 0.10, 0.10]:
        raise ValueError("The frozen 15% 80/10/10 masking policy changed.")
    if config["optimization"]["epochs"] != 10 or config["optimization"]["early_stopping"]:
        raise ValueError("Phase 8R5B must freeze the epoch-10 checkpoint without early stopping.")
    execution = config["execution"]
    if execution["p5_allowed"] or execution["labels_allowed"] or execution["supervised_training"]:
        raise ValueError("P5, labels, and supervised training are prohibited.")
    if execution["device"] != "cuda:0" or not execution["amp"] or execution["physical_batch_size"] != 8:
        raise ValueError("The approved CUDA/AMP/batch-size policy changed.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in config["frozen_manifests"]:
        observed.update(verify_frozen_manifest(manifest))
    return observed


def verify_cuda(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Phase 8R5B will not fall back to CPU.")
    device_name = torch.cuda.get_device_name(0)
    if device_name != config["execution"]["expected_gpu"]:
        raise RuntimeError(f"Expected {config['execution']['expected_gpu']}, found {device_name}.")
    return {
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": True,
        "device_name": device_name,
        "amp": True,
        "physical_batch_size": 8,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_entry(run: dict[str, Any], corpus: PreparedCorpus) -> dict[str, Any]:
    path = Path(run["checkpoint_path"])
    return {
        "path": str(path),
        "fold": run["fold"],
        "seed": run["seed"],
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "vocabulary_hash": corpus.vocabulary_hash,
        "corpus_definition": "+".join(corpus.historical_periods),
        "final_training_loss": run["final_training_loss"],
        "final_masked_token_accuracy": run["final_masked_token_accuracy"],
        "epoch": 10,
    }


def _write_csv(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(output / name, index=False)


def _write_outputs(
    config: dict[str, Any],
    progress: dict[str, Any],
    *,
    status: str,
    started_at: float,
) -> dict[str, Any]:
    output = config["output_dir"]
    _write_csv(output, "pretraining_runs.csv", progress["runs"])
    _write_csv(output, "epoch_metrics.csv", progress["epochs"])
    _write_csv(output, "resource_usage.csv", progress["resources"])
    _atomic_json(output / "checkpoint_manifest.json", {"checkpoints": progress["checkpoints"]})
    _atomic_json(output / "vocabulary_manifest.json", {"vocabularies": progress["vocabularies"]})
    summed = sum(float(item["runtime_seconds"]) for item in progress["runs"])
    estimated = float(progress["phase8r5b0_estimated_three_seed_seconds"])
    fold_summaries: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        fold_runs = [item for item in progress["runs"] if item["fold"] == fold]
        fold_epochs = [item for item in progress["epochs"] if item["fold"] == fold]
        if not fold_runs:
            continue
        losses = np.asarray([item["final_training_loss"] for item in fold_runs], dtype=float)
        accuracies = np.asarray([item["final_masked_token_accuracy"] for item in fold_runs], dtype=float)
        epoch_one = [item for item in fold_epochs if item["epoch"] == 1]
        epoch_ten = [item for item in fold_epochs if item["epoch"] == 10]
        fold_summaries.append({
            "fold": fold,
            "seed_count": len(fold_runs),
            "final_loss_mean": float(losses.mean()),
            "final_loss_std": float(losses.std(ddof=0)),
            "final_accuracy_mean": float(accuracies.mean()),
            "final_accuracy_std": float(accuracies.std(ddof=0)),
            "epoch1_loss_mean": float(np.mean([item["mlm_training_loss"] for item in epoch_one])),
            "epoch10_loss_mean": float(np.mean([item["mlm_training_loss"] for item in epoch_ten])),
            "epoch1_accuracy_mean": float(np.mean([item["masked_token_top1_accuracy"] for item in epoch_one])),
            "epoch10_accuracy_mean": float(np.mean([item["masked_token_top1_accuracy"] for item in epoch_ten])),
        })
    results = {
        "phase": "8R5B",
        "status": status,
        "scope": "masked-opcode self-supervised pretraining only",
        "h8r5_evaluated": False,
        "h8r5_supported": None,
        "p5_loaded": False,
        "ponzi_labels_loaded": False,
        "supervised_classifier_trained": False,
        "frozen_artifact_verification": progress["frozen_artifact_verification"],
        "cuda": progress["cuda"],
        "optimizer": config["optimization"],
        "masking": config["masking"],
        "architecture": config["model"],
        "sequence": config["sequence"],
        "fold_corpora": progress["vocabularies"],
        "runs_completed": len(progress["runs"]),
        "runs_expected": 9,
        "total_gpu_wall_clock_seconds": summed,
        "current_invocation_wall_clock_seconds": time.perf_counter() - started_at,
        "total_summed_run_seconds": summed,
        "phase8r5b0_estimated_three_seed_seconds": estimated,
        "actual_to_estimated_run_time_ratio": summed / estimated if estimated else None,
        "checkpoints": progress["checkpoints"],
        "fold_seed_summaries": fold_summaries,
        "embedding_readiness": config["embedding_readiness"],
        "checkpoint_retraining_bit_exact_claimed": False,
    }
    _atomic_json(output / "phase8r5b_results.json", results)
    report_lines = [
        "# Phase 8R5B Masked-Opcode Pretraining",
        "",
        f"Status: **{status}** ({len(progress['runs'])}/9 runs complete).",
        "",
        "This phase trains only fold-local masked-opcode encoders. It does not load Ponzi labels, evaluate P5, or train a classifier. H8R5 remains unevaluated until Phase 8R5C.",
        "",
        "## Frozen Policy",
        "",
        "- Folds: P0+P1 -> P2, P0-P2 -> P3, P0-P3 -> P4; future periods are excluded from each vocabulary and corpus.",
        "- Seeds: 42, 123, 2026; every fold/seed starts from a fresh model and optimizer.",
        "- Transformer: embedding 128, 2 layers, 4 heads, feed-forward 256, dropout 0.1.",
        "- Input: deterministic 1,024-token chunks; 15% masking with the fixed 80/10/10 replacement policy.",
        "- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-2, exactly 10 epochs.",
        "- Execution: CUDA AMP, physical batch size 8, no CPU fallback.",
        "",
        "## Corpus And Vocabulary",
        "",
        "| Fold | Historical periods | Contracts | Chunks | Non-padding tokens | Vocabulary | Vocabulary SHA-256 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in progress["vocabularies"]:
        report_lines.append(
            f"| {item['fold']} | {item['historical_periods']} | {item['contracts']} | {item['chunks']} | {item['non_padding_tokens']} | {item['vocabulary_size']} | `{item['vocabulary_hash']}` |"
        )
    report_lines.extend(["", "## Completed Runs", "", "| Fold | Seed | Final loss | Final top-1 accuracy | Runtime (min) | Peak VRAM MiB |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for run in progress["runs"]:
        report_lines.append(
            f"| {run['fold']} | {run['seed']} | {run['final_training_loss']:.6f} | {run['final_masked_token_accuracy']:.6f} | {run['runtime_seconds'] / 60:.2f} | {run['peak_reserved_vram_mib']:.1f} |"
        )
    report_lines.extend(["", "## Seed Variability And Learning Trend", "", "| Fold | Final loss mean +/- SD | Final accuracy mean +/- SD | Epoch 1 -> 10 loss | Epoch 1 -> 10 accuracy |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in fold_summaries:
        report_lines.append(
            f"| {item['fold']} | {item['final_loss_mean']:.6f} +/- {item['final_loss_std']:.6f} | {item['final_accuracy_mean']:.6f} +/- {item['final_accuracy_std']:.6f} | {item['epoch1_loss_mean']:.6f} -> {item['epoch10_loss_mean']:.6f} | {item['epoch1_accuracy_mean']:.6f} -> {item['epoch10_accuracy_mean']:.6f} |"
        )
    report_lines.extend([
        "", "## Resource Accounting", "",
        f"- Summed completed-run time: {summed / 60:.2f} minutes.",
        f"- Phase 8R5B0 estimate for all nine runs: {estimated / 60:.2f} minutes.",
        f"- Actual/estimated ratio so far: {(summed / estimated if estimated else 0):.3f}.",
        "- CUDA determinism controls are enabled, but bit-identical GPU retraining is not claimed.",
        "", "## Embedding Readiness", "",
        "The frozen encoder exposes final-layer per-token hidden states. Phase 8R5C is intended to use padding-aware mean pooling for a chunk representation. No supervised embeddings or classifier were produced here.",
    ])
    (output / "phase8r5b_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return results


def run_phase8r5b(config_path: Path = DEFAULT_CONFIG_PATH, *, max_new_runs: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    started_at = time.perf_counter()
    frozen = verify_frozen_inputs(config)
    cuda = verify_cuda(config)
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "phase8r5b_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["config_sha256"] != config["config_sha256"]:
            raise RuntimeError("Cannot resume Phase 8R5B with a changed configuration.")
        for checkpoint in progress["checkpoints"]:
            path = Path(checkpoint["path"])
            if not path.is_file() or sha256_file(path) != checkpoint["sha256"]:
                raise RuntimeError(f"Completed checkpoint changed: {path}")
    else:
        benchmark_results = json.loads(config["inputs"]["phase8r5b0_results"].read_text(encoding="utf-8"))
        progress = {
            "config_sha256": config["config_sha256"],
            "frozen_artifact_verification": {"verified": True, "file_count": len(frozen), "hashes": frozen},
            "cuda": cuda,
            "phase8r5b0_estimated_three_seed_seconds": float(benchmark_results["three_seed_ten_epoch_hours"]) * 3600,
            "vocabularies": [], "runs": [], "epochs": [], "resources": [], "checkpoints": [],
        }
    frame = load_p0_p4_sequences(config["inputs"]["dataset"], config["inputs"]["period_definitions"])
    completed = {item["run_id"] for item in progress["runs"]}
    newly_completed = 0
    preprocessing_hashes = {
        "raw_dataset_sha256": sha256_file(config["inputs"]["dataset"]),
        "period_definitions_sha256": sha256_file(config["inputs"]["period_definitions"]),
        "phase8r5a_config_sha256": sha256_file(config["inputs"]["phase8r5a_config"]),
        "phase8r5b0_config_sha256": sha256_file(config["inputs"]["phase8r5b0_config"]),
        "phase8r5b_config_sha256": config["config_sha256"],
    }

    for fold, periods in EXPECTED_FOLDS.items():
        corpus = prepare_fold_corpus(frame, fold=fold, historical_periods=periods, chunk_length=1024)
        if not any(item["fold"] == fold for item in progress["vocabularies"]):
            progress["vocabularies"].append({
                "fold": fold,
                "historical_periods": "+".join(periods),
                "future_period_excluded": fold,
                "contracts": corpus.contract_count,
                "chunks": int(len(corpus.chunks)),
                "non_padding_tokens": corpus.non_padding_tokens,
                "padding_tokens": corpus.padding_tokens,
                "mean_chunks_per_contract": corpus.mean_chunks_per_contract,
                "max_chunks_per_contract": corpus.max_chunks_per_contract,
                "vocabulary_size": len(corpus.vocabulary.token_to_id),
                "ordinary_opcode_vocabulary_size": len(corpus.vocabulary.token_to_id) - len(SPECIAL_TOKENS),
                "vocabulary_hash": corpus.vocabulary_hash,
                "vocabulary_mapping": corpus.vocabulary.token_to_id,
            })
            _atomic_json(progress_path, progress)
        for seed in EXPECTED_SEEDS:
            run_id = f"{fold}_seed_{seed}"
            if run_id in completed:
                continue
            if max_new_runs is not None and newly_completed >= max_new_runs:
                _atomic_json(progress_path, progress)
                return _write_outputs(config, progress, status="partial_resumable", started_at=started_at)
            checkpoint_path = config["checkpoint_dir"] / fold.lower() / f"opcode_mlm_{fold.lower()}_seed{seed}_epoch10.pt"
            epochs, run, resource = train_pretraining_run(
                corpus,
                seed=seed,
                config=config,
                checkpoint_path=checkpoint_path,
                preprocessing_hashes=preprocessing_hashes,
            )
            checkpoint = _checkpoint_entry(run, corpus)
            resource["checkpoint_sha256"] = checkpoint["sha256"]
            progress["epochs"].extend(epochs)
            progress["runs"].append(run)
            progress["resources"].append(resource)
            progress["checkpoints"].append(checkpoint)
            completed.add(run_id)
            newly_completed += 1
            _atomic_json(progress_path, progress)
            _write_outputs(config, progress, status="running", started_at=started_at)
    _atomic_json(progress_path, progress)
    return _write_outputs(config, progress, status="complete", started_at=started_at)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-new-runs", type=int)
    args = parser.parse_args()
    results = run_phase8r5b(args.config, max_new_runs=args.max_new_runs)
    print(json.dumps({"status": results["status"], "runs_completed": results["runs_completed"]}, indent=2))


if __name__ == "__main__":
    main()
