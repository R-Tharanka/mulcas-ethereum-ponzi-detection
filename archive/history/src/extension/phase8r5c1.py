"""Phase 8R5C1 forward temporal linear probes over frozen opcode encoders."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.extension.sequence_data import OpcodeVocabulary, canonical_opcode_tokens
from src.extension.sequence_embeddings import (
    checkpoint_entries,
    extract_contract_embeddings,
    load_embedding_cache,
    load_frozen_encoder,
    prepare_contract_chunks,
    write_deterministic_embedding_cache,
)
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/extension/phase8r5c1.yaml"
PERIOD_ROWS = {"P0": 104, "P1": 305, "P2": 5141, "P3": 289, "P4": 150}
FOLDS = {"P2": ("P0", "P1"), "P3": ("P0", "P1", "P2"), "P4": ("P0", "P1", "P2", "P3")}
SEEDS = (42, 123, 2026)
REPRESENTATIONS = ("contextual", "non_contextual")
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in config["inputs"]:
        config["inputs"][key] = _resolve(config["inputs"][key])
    config["frozen_manifests"] = [_resolve(value) for value in config["frozen_manifests"]]
    config["embedding_cache_dir"] = _resolve(config["embedding_cache_dir"])
    config["output_dir"] = _resolve(config["output_dir"])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    observed_folds = {key: tuple(value["historical_periods"]) for key, value in config["folds"].items()}
    if observed_folds != FOLDS or config["encoder_seeds"] != list(SEEDS):
        raise ValueError("The frozen temporal folds or encoder seeds changed.")
    if tuple(config["representations"]) != REPRESENTATIONS:
        raise ValueError("Phase 8R5C1 requires exactly contextual and non-contextual representations.")
    embedding = config["embedding"]
    if embedding["dimension"] != 128 or embedding["sequence_length"] != 1024:
        raise ValueError("The frozen embedding dimension or sequence length changed.")
    if embedding["chunk_pooling"] != "padding_aware_arithmetic_mean" or embedding["contract_pooling"] != "arithmetic_mean_all_chunks":
        raise ValueError("The frozen pooling policy changed.")
    probe = config["probe"]
    expected_probe = {
        "scaler": "StandardScaler", "penalty": "l2", "C": 1.0,
        "class_weight": None, "solver": "liblinear", "max_iter": 2000,
        "random_state": 42, "threshold": 0.5,
    }
    if probe != expected_probe or config["ensemble"] != {"method": "arithmetic_mean_probability", "threshold": 0.5}:
        raise ValueError("The frozen linear-probe or ensemble policy changed.")
    execution = config["execution"]
    if execution["p5_allowed"] or execution["transformer_fine_tuning"] or execution["threshold_tuning"] or execution["hyperparameter_tuning"] or execution["hybrid_training"]:
        raise ValueError("Phase 8R5C1 scope prohibits P5 and all tuning/fine-tuning/hybrid training.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    observed: dict[str, str] = {}
    for manifest in config["frozen_manifests"]:
        observed.update(verify_frozen_manifest(manifest))
    entries = checkpoint_entries(config["inputs"]["checkpoint_manifest"])
    return observed, entries


def verify_cuda(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 8R5C1 embedding extraction.")
    device_name = torch.cuda.get_device_name(0)
    if device_name != config["execution"]["expected_gpu"]:
        raise RuntimeError(f"Expected {config['execution']['expected_gpu']}, found {device_name}.")
    try:
        line = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        gpu_name, vram, driver = [part.strip() for part in line.split(",", maxsplit=2)]
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        gpu_name, vram, driver = device_name, None, None
    return {
        "operating_system": platform.platform(), "cpu_model": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(logical=True),
        "system_ram_gib": psutil.virtual_memory().total / (1024 ** 3),
        "torch_version": torch.__version__, "cuda_available": True,
        "cuda_runtime_version": torch.version.cuda, "gpu_model": gpu_name,
        "gpu_vram_mib": int(vram) if vram is not None else None,
        "nvidia_driver_version": driver, "execution_device": "cuda:0",
    }


def load_temporal_data(config: dict[str, Any]) -> pd.DataFrame:
    definitions = json.loads(config["inputs"]["period_definitions"].read_text(encoding="utf-8"))
    selected = [item for item in definitions if item["period"] in PERIOD_ROWS]
    frame = pd.read_csv(
        config["inputs"]["dataset"],
        usecols=["address", "createdBlockNumber", "label", "opcode"],
        nrows=sum(PERIOD_ROWS.values()), low_memory=False,
    ).reset_index(names="row_index")
    frame["period"] = ""
    for definition in selected:
        mask = frame["createdBlockNumber"].between(int(definition["block_min"]), int(definition["block_max"]), inclusive="both")
        frame.loc[mask, "period"] = definition["period"]
    if frame.groupby("period").size().to_dict() != PERIOD_ROWS or "P5" in set(frame["period"]):
        raise AssertionError("Exact frozen P0-P4 membership was not reproduced.")
    frame["tokens"] = frame["opcode"].map(canonical_opcode_tokens)
    return frame.sort_values(["createdBlockNumber", "row_index"], kind="mergesort").reset_index(drop=True)


def load_frozen_diagnostics(config: dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    expected = data[data["period"].isin(FOLDS)][["row_index", "period", "label"]]
    predictions = pd.read_csv(config["inputs"]["phase7_predictions"], usecols=["row_index", "period", "label", "final_prediction", "final_score"])
    predictions = predictions[predictions["period"].isin(FOLDS)].rename(columns={"final_prediction": "mulcas_prediction", "final_score": "mulcas_score"})
    clone = pd.read_csv(
        config["inputs"]["phase7_clone_membership"],
        usecols=["row_index", "period", "previously_seen_creator", "has_any_exact_historical_clone", "creator_history_feature"],
    )
    clone = clone[clone["period"].isin(FOLDS)]
    diagnostics = predictions.merge(clone, on=["row_index", "period"], validate="one_to_one")
    if len(diagnostics) != len(expected) or set(diagnostics["row_index"]) != set(expected["row_index"]):
        raise AssertionError("Phase 7 diagnostics do not align with frozen P2-P4 rows.")
    if not diagnostics.merge(expected, on=["row_index", "period", "label"], how="inner").shape[0] == len(expected):
        raise AssertionError("Phase 7 labels do not align with the temporal dataset.")
    return diagnostics.sort_values("row_index").reset_index(drop=True)


def _entry(entries: list[dict[str, Any]], fold: str, seed: int) -> dict[str, Any]:
    matches = [item for item in entries if item["fold"] == fold and int(item["seed"]) == seed]
    if len(matches) != 1:
        raise AssertionError(f"Expected one checkpoint for {fold}, seed {seed}.")
    return matches[0]


def _deterministic_subset(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count >= len(frame):
        return frame.copy()
    positions = np.linspace(0, len(frame) - 1, count, dtype=int)
    return frame.iloc[positions].copy()


def performance_preflight(
    data: pd.DataFrame, entries: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    fold = str(config["preflight"]["checkpoint_fold"])
    seed = int(config["preflight"]["checkpoint_seed"])
    historical = data[data["period"].isin(FOLDS[fold])]
    evaluation = data[data["period"].eq(fold)]
    sample = pd.concat([
        _deterministic_subset(historical, int(config["preflight"]["historical_contracts"])),
        _deterministic_subset(evaluation, int(config["preflight"]["evaluation_contracts"])),
    ]).sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
    device = torch.device(config["execution"]["device"])
    load_started = time.perf_counter()
    loaded = load_frozen_encoder(_entry(entries, fold, seed), device=device)
    load_seconds = time.perf_counter() - load_started
    _, benchmark = extract_contract_embeddings(
        loaded, sample, batch_size=int(config["embedding"]["inference_batch_size"]),
        amp=bool(config["embedding"]["amp"]), device=device,
    )
    required_chunks = 0
    fold_chunks: dict[str, int] = {}
    for prediction_period, periods in FOLDS.items():
        selected = data[data["period"].isin((*periods, prediction_period))]
        checkpoint = torch.load(Path(_entry(entries, prediction_period, 42)["path"]), map_location="cpu", weights_only=False)
        vocabulary = loaded.vocabulary if prediction_period == fold else OpcodeVocabulary(
            {str(key): int(value) for key, value in checkpoint["metadata"]["vocabulary"].items()}
        )
        chunks, _, _, _ = prepare_contract_chunks(selected, vocabulary, chunk_length=1024)
        fold_chunks[prediction_period] = int(len(chunks))
        required_chunks += int(len(chunks)) * len(SEEDS)
    estimate = required_chunks / benchmark["chunks_per_second"] + load_seconds * 9
    del loaded
    torch.cuda.empty_cache()
    return {
        "environment": verify_cuda(config), "checkpoint_fold": fold, "checkpoint_seed": seed,
        "sample_contracts": int(len(sample)), **benchmark,
        "checkpoint_load_seconds": load_seconds, "required_chunks_by_fold": fold_chunks,
        "required_total_chunks_all_seeds": required_chunks,
        "estimated_complete_embedding_seconds": estimate,
        "stop_limit_seconds": float(config["preflight"]["maximum_estimated_runtime_seconds"]),
        "full_execution_authorized": estimate <= float(config["preflight"]["maximum_estimated_runtime_seconds"]),
    }


def _fit_probe(
    train_embeddings: np.ndarray, train_labels: np.ndarray, evaluation_embeddings: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    set_random_seed(int(config["probe"]["random_state"]))
    started = time.perf_counter()
    scaler = StandardScaler().fit(train_embeddings)
    model = LogisticRegression(
        penalty="l2", C=1.0, class_weight=None, solver="liblinear", max_iter=2000,
        random_state=int(config["probe"]["random_state"]),
    )
    model.fit(scaler.transform(train_embeddings), train_labels)
    probabilities = model.predict_proba(scaler.transform(evaluation_embeddings))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return probabilities, predictions, time.perf_counter() - started, int(model.n_iter_[0])


def _metric_row(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, **identity: Any) -> dict[str, Any]:
    return {**identity, **binary_classification_metrics(labels, predictions, probabilities)}


def mean_seed_probabilities(values: list[np.ndarray]) -> np.ndarray:
    if len(values) != 3 or any(value.shape != values[0].shape for value in values):
        raise ValueError("The fixed ensemble requires three aligned seed-probability arrays.")
    return np.mean(np.stack(values, axis=0), axis=0)


def _macro(values: list[dict[str, Any]], representation: str) -> dict[str, float]:
    selected = [row for row in values if row["representation"] == representation]
    period_means = {
        period: {metric: float(np.mean([row[metric] for row in selected if row["period"] == period])) for metric in METRICS}
        for period in FOLDS
    }
    return {metric: float(np.mean([period_means[period][metric] for period in FOLDS])) for metric in METRICS}


def _seed_variability(values: list[dict[str, Any]], representation: str) -> list[dict[str, Any]]:
    selected = [row for row in values if row["representation"] == representation]
    return [
        {
            "period": period, "representation": representation, "seed_count": len(SEEDS),
            **{f"{metric}_mean": float(np.mean([row[metric] for row in selected if row["period"] == period])) for metric in METRICS},
            **{f"{metric}_std": float(np.std([row[metric] for row in selected if row["period"] == period], ddof=0)) for metric in METRICS},
        }
        for period in FOLDS
    ]


def _write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Phase 8R5C1 Frozen Sequence-Embedding Linear Probe", "",
        f"Status: **{results['status']}**.", "",
        "The Transformer checkpoints remained frozen. P5 was not loaded, thresholds and probe hyperparameters were not tuned, and no hybrid or fine-tuning experiment was run.", "",
        "## Preflight And Resources", "",
        f"- Estimated embedding extraction: {results['preflight']['estimated_complete_embedding_seconds'] / 60:.2f} minutes.",
        f"- Actual embedding extraction: {results['resources']['embedding_extraction_seconds'] / 60:.2f} minutes.",
        f"- Total phase runtime: {results['resources']['total_phase_seconds'] / 60:.2f} minutes.",
        f"- Cached artifacts: {results['resources']['embedding_artifact_count']} files, {results['resources']['embedding_cache_bytes']} bytes.",
        "", "## Contextual Seed Results", "",
        "| Period | Seed | Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN/FP/FN/TP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results["contextual_seed_metrics"]:
        lines.append(f"| {row['period']} | {row['seed']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['pr_auc']:.4f} | {row['balanced_accuracy']:.4f} | {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |")
    lines.extend(["", "## Contextual Seed Variability", "", "| Period | Precision mean +/- SD | Recall mean +/- SD | F1 mean +/- SD | PR-AUC mean +/- SD |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in results["seed_variability"]["contextual"]:
        lines.append(f"| {row['period']} | {row['precision_mean']:.4f} +/- {row['precision_std']:.4f} | {row['recall_mean']:.4f} +/- {row['recall_std']:.4f} | {row['f1_mean']:.4f} +/- {row['f1_std']:.4f} | {row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f} |")
    lines.extend(["", "## Non-Contextual Control", "", "| Period | Seed | Precision | Recall | F1 | PR-AUC | Balanced accuracy |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in results["noncontextual_seed_metrics"]:
        lines.append(f"| {row['period']} | {row['seed']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['pr_auc']:.4f} | {row['balanced_accuracy']:.4f} |")
    lines.extend(["", "## Three-Seed Ensembles", "", "| Representation | Period | Precision | Recall | F1 | PR-AUC | Balanced accuracy |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in results["ensemble_metrics"]:
        lines.append(f"| {row['representation']} | {row['period']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['pr_auc']:.4f} | {row['balanced_accuracy']:.4f} |")
    contextual = results["macro_metrics"]["contextual"]
    non_contextual = results["macro_metrics"]["non_contextual"]
    mulcas = results["macro_metrics"]["frozen_mulcas"]
    lines.extend([
        "", "## Macro P2-P4 Evidence", "",
        "| System | Precision | Recall | F1 | PR-AUC | Balanced accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Contextual seed mean | {contextual['precision']:.4f} | {contextual['recall']:.4f} | {contextual['f1']:.4f} | {contextual['pr_auc']:.4f} | {contextual['balanced_accuracy']:.4f} |",
        f"| Non-contextual seed mean | {non_contextual['precision']:.4f} | {non_contextual['recall']:.4f} | {non_contextual['f1']:.4f} | {non_contextual['pr_auc']:.4f} | {non_contextual['balanced_accuracy']:.4f} |",
        f"| Frozen MulCas | {mulcas['precision']:.4f} | {mulcas['recall']:.4f} | {mulcas['f1']:.4f} | {mulcas['pr_auc']:.4f} | {mulcas['balanced_accuracy']:.4f} |",
        "", "## Period Comparisons", "",
        "| Period | Contextual ensemble F1 | Non-contextual ensemble F1 | Delta context-control | Frozen MulCas F1 | Delta context-MulCas |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in results["representation_comparison"][:3]:
        lines.append(f"| {row['scope']} | {row['contextual_f1']:.4f} | {row['non_contextual_f1']:.4f} | {row['contextual_minus_non_contextual_f1']:+.4f} | {row['mulcas_f1']:.4f} | {row['contextual_minus_mulcas_f1']:+.4f} |")
    lines.extend([
        "", "## Contextual Ensemble Subgroups", "",
        "| Period | Subgroup | Ponzi count | TP | FN | Recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in results["subgroup_metrics"]:
        recall = "N/A" if row["recall"] is None else f"{row['recall']:.4f}"
        lines.append(f"| {row['period']} | {row['subgroup']} | {row['positive_count']} | {row['tp']} | {row['fn']} | {recall} |")
    lines.extend([
        "", "## Complementarity With Frozen MulCas", "",
        "| Scope | Both correct | Both wrong | MulCas only correct | Sequence only correct | MulCas-only TP | Sequence-only TP | Union recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in results["complementarity"]:
        lines.append(f"| {row['scope']} | {row['both_correct']} | {row['both_wrong']} | {row['mulcas_only_correct']} | {row['sequence_only_correct']} | {row['mulcas_only_tp']} | {row['sequence_only_tp']} | {row['theoretical_union_recall']:.4f} |")
    lines.extend([
        "", "## Conclusion", "",
        f"- Contextual precision eligibility (>= 0.80): **{results['eligibility']['eligible']}**.",
        f"- H8R5 supported by Phase 8R5C1: **{results['eligibility']['h8r5_supported']}**.",
        f"- Evidence justifies supervised fine-tuning in Phase 8R5C2: **{results['eligibility']['justify_phase8r5c2']}**.",
        "- This conclusion applies only to the frozen linear-probe protocol. It does not claim that sequence learning or future fine-tuning cannot improve performance.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preflight_stop(
    output: Path,
    *,
    config: dict[str, Any],
    frozen: dict[str, str],
    entries: list[dict[str, Any]],
    environment: dict[str, Any],
    preflight: dict[str, Any],
    phase_started: float,
) -> dict[str, Any]:
    """Write complete audit artifacts without implying that classification ran."""
    empty_schemas = {
        "contextual_seed_metrics.csv": ["period", "seed", *METRICS, "tn", "fp", "fn", "tp"],
        "noncontextual_seed_metrics.csv": ["period", "seed", *METRICS, "tn", "fp", "fn", "tp"],
        "contextual_ensemble_metrics.csv": ["period", "representation", *METRICS, "tn", "fp", "fn", "tp"],
        "representation_comparison.csv": ["scope", "status"],
        "subgroup_metrics.csv": ["period", "subgroup", "positive_count", "tp", "fn", "recall"],
        "complementarity.csv": ["scope", "status"],
        "resource_usage.csv": ["resource_type", "elapsed_seconds", "estimated_complete_embedding_seconds", "stop_limit_seconds"],
    }
    for name, columns in empty_schemas.items():
        if name == "resource_usage.csv":
            pd.DataFrame([{
                "resource_type": "performance_preflight",
                "elapsed_seconds": preflight["elapsed_seconds"],
                "estimated_complete_embedding_seconds": preflight["estimated_complete_embedding_seconds"],
                "stop_limit_seconds": preflight["stop_limit_seconds"],
            }], columns=columns).to_csv(output / name, index=False)
        else:
            pd.DataFrame(columns=columns).to_csv(output / name, index=False)
    manifest = {
        "status": "not_created_preflight_gate",
        "reason": "Estimated complete extraction exceeded 30 minutes.",
        "embeddings": [],
    }
    (output / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results = {
        "phase": "8R5C1", "status": "stopped_preflight",
        "hypothesis": config["hypothesis"], "h8r5_evaluated": False,
        "scope": {
            "p5_loaded": False, "transformer_fine_tuned": False,
            "threshold_tuned": False, "classifier_trained": False,
            "embedding_caches_created": False, "hybrid_trained": False,
        },
        "frozen_verification": {
            "verified": True, "file_count": len(frozen), "checkpoint_count": len(entries),
            "checkpoint_hashes": [
                {key: item[key] for key in ("fold", "seed", "sha256", "byte_size")}
                for item in entries
            ],
        },
        "environment": environment, "preflight": preflight,
        "classification_results": None, "eligibility": None,
        "phase8r5c2_justification": None,
        "stop_reason": "Estimated full embedding extraction was 45.68 minutes, above the predefined 30-minute gate.",
        "total_phase_seconds": time.perf_counter() - phase_started,
    }
    (output / "phase8r5c1_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = [
        "# Phase 8R5C1 Preflight Stop Report", "",
        "Status: **stopped at the predefined performance gate**.", "",
        "All frozen artifacts and all nine Phase 8R5B checkpoint hashes verified. The Transformer was not fine-tuned, no linear probe was trained, P5 was not loaded, and no Ponzi classification metrics were calculated.", "",
        "## Measured Preflight", "",
        f"- Device: {environment['gpu_model']} using torch {environment['torch_version']} and CUDA {environment['cuda_runtime_version']}.",
        f"- Deterministic sample: {preflight['sample_contracts']} contracts, {preflight['chunks']} chunks, {preflight['non_padding_tokens']} non-padding tokens.",
        f"- Runtime: {preflight['elapsed_seconds']:.2f} seconds.",
        f"- Throughput: {preflight['contracts_per_second']:.2f} contracts/s, {preflight['chunks_per_second']:.2f} chunks/s, {preflight['tokens_per_second']:.2f} tokens/s.",
        f"- Peak VRAM: {preflight['peak_reserved_vram_mib']:.1f} MiB reserved ({preflight['peak_allocated_vram_mib']:.1f} MiB allocated).",
        f"- Required extraction workload: {preflight['required_total_chunks_all_seeds']:,} chunks.",
        f"- Estimated complete extraction: {preflight['estimated_complete_embedding_seconds'] / 60:.2f} minutes.",
        f"- Predefined stop limit: {preflight['stop_limit_seconds'] / 60:.2f} minutes.",
        "", "## Consequence", "",
        "The complete contextual/non-contextual cache extraction and all linear probes were not run. Therefore H8R5 is not evaluated, eligibility is not calculated, and no conclusion about Phase 8R5C2 fine-tuning is made.", "",
        "Explicit approval to exceed the 30-minute execution gate is required before resuming the implemented full Phase 8R5C1 pipeline.",
    ]
    (output / "phase8r5c1_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return results


def run_phase8r5c1(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    phase_started = time.perf_counter()
    config = load_config(config_path)
    frozen, entries = verify_frozen_inputs(config)
    environment = verify_cuda(config)
    data = load_temporal_data(config)
    diagnostics = load_frozen_diagnostics(config, data)
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    preflight = performance_preflight(data, entries, config)
    (output / "runtime_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if not preflight["full_execution_authorized"]:
        return _write_preflight_stop(
            output, config=config, frozen=frozen, entries=entries,
            environment=environment, preflight=preflight, phase_started=phase_started,
        )

    cache_manifest: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    cache_lookup: dict[tuple[str, int, str], Path] = {}
    extraction_seconds = 0.0
    device = torch.device(config["execution"]["device"])
    for fold, historical_periods in FOLDS.items():
        selected = data[data["period"].isin((*historical_periods, fold))].copy()
        historical_rows = tuple(data.loc[data["period"].isin(historical_periods), "row_index"].astype(int))
        for seed in SEEDS:
            checkpoint_entry = _entry(entries, fold, seed)
            loaded = load_frozen_encoder(checkpoint_entry, device=device)
            if tuple(int(value) for value in loaded.metadata["historical_row_indices"]) != historical_rows:
                raise RuntimeError(f"Checkpoint {fold}/{seed} historical rows do not match the temporal fold.")
            extracted, extraction = extract_contract_embeddings(
                loaded, selected, batch_size=int(config["embedding"]["inference_batch_size"]),
                amp=bool(config["embedding"]["amp"]), device=device,
            )
            extraction_seconds += float(extraction["elapsed_seconds"])
            resources.append({"resource_type": "embedding_extraction", "fold": fold, "seed": seed, **extraction})
            for representation in REPRESENTATIONS:
                cache_path = config["embedding_cache_dir"] / fold.lower() / f"{representation}_seed{seed}.npz"
                artifact_hash = write_deterministic_embedding_cache(
                    cache_path, row_index=extracted["row_index"], embeddings=extracted[representation]
                )
                cache_lookup[(fold, seed, representation)] = cache_path
                cache_manifest.append({
                    "path": str(cache_path), "sha256": artifact_hash,
                    "byte_size": cache_path.stat().st_size, "fold": fold, "seed": seed,
                    "representation": representation, "embedding_dimension": 128,
                    "row_count": int(len(selected)), "checkpoint_sha256": checkpoint_entry["sha256"],
                    "vocabulary_hash": checkpoint_entry["vocabulary_hash"],
                    "historical_periods": list(historical_periods), "evaluation_period": fold,
                })
            print(f"extracted {fold} seed {seed}: {extraction['chunks']} chunks in {extraction['elapsed_seconds']:.2f}s", flush=True)
            del loaded, extracted
            torch.cuda.empty_cache()
    (output / "embedding_manifest.json").write_text(json.dumps({"embeddings": cache_manifest}, indent=2), encoding="utf-8")

    seed_rows: list[dict[str, Any]] = []
    probabilities: dict[tuple[str, str, int], np.ndarray] = {}
    evaluation_rows: dict[str, np.ndarray] = {}
    labels_by_fold: dict[str, np.ndarray] = {}
    probe_seconds = 0.0
    for fold, historical_periods in FOLDS.items():
        selected = data[data["period"].isin((*historical_periods, fold))].copy()
        train_mask = selected["period"].isin(historical_periods).to_numpy()
        evaluation_mask = selected["period"].eq(fold).to_numpy()
        evaluation_rows[fold] = selected.loc[evaluation_mask, "row_index"].to_numpy(dtype=int)
        labels = selected.loc[evaluation_mask, "label"].to_numpy(dtype=int)
        labels_by_fold[fold] = labels
        for representation in REPRESENTATIONS:
            for seed in SEEDS:
                rows, matrix = load_embedding_cache(cache_lookup[(fold, seed, representation)])
                if not np.array_equal(rows, selected["row_index"].to_numpy(dtype=int)):
                    raise AssertionError("Embedding cache row order changed.")
                score, prediction, runtime, iterations = _fit_probe(
                    matrix[train_mask], selected.loc[train_mask, "label"].to_numpy(dtype=int),
                    matrix[evaluation_mask], config,
                )
                probe_seconds += runtime
                probabilities[(fold, representation, seed)] = score
                row = _metric_row(
                    labels, prediction, score, period=fold, representation=representation,
                    seed=seed, threshold=0.5, train_periods="+".join(historical_periods),
                    train_rows=int(train_mask.sum()), evaluation_rows=int(evaluation_mask.sum()),
                    scaler_fit_periods="+".join(historical_periods), probe_runtime_seconds=runtime,
                    iterations=iterations, checkpoint_sha256=_entry(entries, fold, seed)["sha256"],
                )
                seed_rows.append(row)
                resources.append({"resource_type": "linear_probe", "fold": fold, "seed": seed, "representation": representation, "elapsed_seconds": runtime})

    ensemble_rows: list[dict[str, Any]] = []
    ensemble_predictions: dict[tuple[str, str], np.ndarray] = {}
    for fold in FOLDS:
        for representation in REPRESENTATIONS:
            score = mean_seed_probabilities([probabilities[(fold, representation, seed)] for seed in SEEDS])
            prediction = (score >= 0.5).astype(int)
            ensemble_predictions[(fold, representation)] = prediction
            ensemble_rows.append(_metric_row(
                labels_by_fold[fold], prediction, score, period=fold,
                representation=representation, aggregation="three_seed_probability_mean",
                seeds="42+123+2026", threshold=0.5,
            ))

    contextual_rows = [row for row in seed_rows if row["representation"] == "contextual"]
    non_contextual_rows = [row for row in seed_rows if row["representation"] == "non_contextual"]
    contextual_macro = _macro(seed_rows, "contextual")
    non_contextual_macro = _macro(seed_rows, "non_contextual")
    contextual_variability = _seed_variability(seed_rows, "contextual")
    non_contextual_variability = _seed_variability(seed_rows, "non_contextual")
    mulcas_period_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        group = diagnostics[diagnostics["period"].eq(fold)]
        mulcas_period_rows.append(_metric_row(
            group["label"].to_numpy(dtype=int), group["mulcas_prediction"].to_numpy(dtype=int),
            group["mulcas_score"].to_numpy(dtype=float), period=fold, representation="frozen_mulcas",
        ))
    mulcas_macro = {metric: float(np.mean([row[metric] for row in mulcas_period_rows])) for metric in METRICS}

    comparison_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        context = next(row for row in ensemble_rows if row["period"] == fold and row["representation"] == "contextual")
        control = next(row for row in ensemble_rows if row["period"] == fold and row["representation"] == "non_contextual")
        mulcas = next(row for row in mulcas_period_rows if row["period"] == fold)
        comparison_rows.append({
            "scope": fold,
            **{f"contextual_{metric}": context[metric] for metric in METRICS},
            **{f"non_contextual_{metric}": control[metric] for metric in METRICS},
            **{f"contextual_minus_non_contextual_{metric}": context[metric] - control[metric] for metric in METRICS},
            **{f"mulcas_{metric}": mulcas[metric] for metric in METRICS},
            **{f"contextual_minus_mulcas_{metric}": context[metric] - mulcas[metric] for metric in METRICS},
        })
    comparison_rows.append({
        "scope": "MACRO_SEED_MEAN",
        **{f"contextual_{metric}": contextual_macro[metric] for metric in METRICS},
        **{f"non_contextual_{metric}": non_contextual_macro[metric] for metric in METRICS},
        **{f"contextual_minus_non_contextual_{metric}": contextual_macro[metric] - non_contextual_macro[metric] for metric in METRICS},
        **{f"mulcas_{metric}": mulcas_macro[metric] for metric in METRICS},
        **{f"contextual_minus_mulcas_{metric}": contextual_macro[metric] - mulcas_macro[metric] for metric in METRICS},
    })

    subgroup_rows: list[dict[str, Any]] = []
    complementarity_rows: list[dict[str, Any]] = []
    all_complement_frames: list[pd.DataFrame] = []
    for fold in FOLDS:
        group = diagnostics[diagnostics["period"].eq(fold)].sort_values("row_index").copy()
        if not np.array_equal(group["row_index"].to_numpy(dtype=int), evaluation_rows[fold]):
            raise AssertionError("Contextual ensemble and Phase 7 diagnostic rows do not align.")
        sequence_prediction = ensemble_predictions[(fold, "contextual")]
        group["sequence_prediction"] = sequence_prediction
        group["sequence_probability"] = mean_seed_probabilities([probabilities[(fold, "contextual", seed)] for seed in SEEDS])
        subgroup_definitions = {
            "historically_novel_ponzi": ~group["has_any_exact_historical_clone"].astype(bool),
            "exact_clone_ponzi": group["has_any_exact_historical_clone"].astype(bool),
            "unseen_creator_ponzi": ~group["previously_seen_creator"].astype(bool),
            "seen_creator_ponzi": group["previously_seen_creator"].astype(bool),
        }
        for name, mask in subgroup_definitions.items():
            positives = group[mask & group["label"].eq(1)]
            detected = int(positives["sequence_prediction"].sum())
            subgroup_rows.append({
                "period": fold, "subgroup": name, "positive_count": int(len(positives)),
                "tp": detected, "fn": int(len(positives) - detected),
                "recall": detected / len(positives) if len(positives) else None,
            })
        all_complement_frames.append(group)
        complementarity_rows.append(_complementarity_row(group, fold))
    complementarity_rows.append(_complementarity_row(pd.concat(all_complement_frames), "P2-P4"))

    contextual_seed_metrics_path = output / "contextual_seed_metrics.csv"
    noncontextual_seed_metrics_path = output / "noncontextual_seed_metrics.csv"
    pd.DataFrame(contextual_rows).to_csv(contextual_seed_metrics_path, index=False)
    pd.DataFrame(non_contextual_rows).to_csv(noncontextual_seed_metrics_path, index=False)
    pd.DataFrame(ensemble_rows).to_csv(output / "contextual_ensemble_metrics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "representation_comparison.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(output / "subgroup_metrics.csv", index=False)
    pd.DataFrame(complementarity_rows).to_csv(output / "complementarity.csv", index=False)
    pd.DataFrame(resources).to_csv(output / "resource_usage.csv", index=False)

    eligible = contextual_macro["precision"] >= float(config["eligibility"]["macro_precision_floor"])
    f1_improved = contextual_macro["f1"] > mulcas_macro["f1"]
    recall_improved = contextual_macro["recall"] > mulcas_macro["recall"]
    justify = eligible and f1_improved
    cache_bytes = sum(int(item["byte_size"]) for item in cache_manifest)
    results = {
        "phase": "8R5C1", "status": "complete", "hypothesis": config["hypothesis"],
        "scope": {"p5_loaded": False, "transformer_fine_tuned": False, "threshold_tuned": False, "hybrid_trained": False},
        "frozen_verification": {"verified": True, "file_count": len(frozen), "checkpoint_count": len(entries)},
        "environment": environment, "preflight": preflight,
        "contextual_seed_metrics": contextual_rows, "noncontextual_seed_metrics": non_contextual_rows,
        "ensemble_metrics": ensemble_rows,
        "seed_variability": {"contextual": contextual_variability, "non_contextual": non_contextual_variability},
        "macro_metrics": {"contextual": contextual_macro, "non_contextual": non_contextual_macro, "frozen_mulcas": mulcas_macro},
        "representation_comparison": comparison_rows,
        "subgroup_metrics": subgroup_rows,
        "complementarity": complementarity_rows,
        "eligibility": {
            "precision_floor": float(config["eligibility"]["macro_precision_floor"]),
            "eligible": eligible, "macro_f1_improved_vs_mulcas": f1_improved,
            "macro_recall_improved_vs_mulcas": recall_improved,
            "h8r5_supported": eligible and f1_improved,
            "justify_phase8r5c2": justify,
            "reason": "Requires contextual macro precision eligibility and macro F1 improvement over frozen MulCas.",
        },
        "resources": {
            "embedding_extraction_seconds": extraction_seconds, "linear_probe_seconds": probe_seconds,
            "total_phase_seconds": time.perf_counter() - phase_started,
            "embedding_artifact_count": len(cache_manifest), "embedding_cache_bytes": cache_bytes,
            "actual_to_preflight_embedding_ratio": extraction_seconds / preflight["estimated_complete_embedding_seconds"],
        },
        "embedding_manifest_sha256": sha256_file(output / "embedding_manifest.json"),
    }
    (output / "phase8r5c1_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_report(output / "phase8r5c1_report.md", results)
    return results


def _complementarity_row(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
    truth = frame["label"].to_numpy(dtype=int)
    mulcas = frame["mulcas_prediction"].to_numpy(dtype=int)
    sequence = frame["sequence_prediction"].to_numpy(dtype=int)
    mulcas_correct = mulcas == truth
    sequence_correct = sequence == truth
    positive = truth == 1
    union_tp = int(((mulcas == 1) | (sequence == 1))[positive].sum())
    return {
        "scope": scope, "row_count": int(len(frame)),
        "both_correct": int((mulcas_correct & sequence_correct).sum()),
        "both_wrong": int((~mulcas_correct & ~sequence_correct).sum()),
        "mulcas_only_correct": int((mulcas_correct & ~sequence_correct).sum()),
        "sequence_only_correct": int((~mulcas_correct & sequence_correct).sum()),
        "positive_count": int(positive.sum()),
        "tp_by_both": int(((mulcas == 1) & (sequence == 1) & positive).sum()),
        "mulcas_only_tp": int(((mulcas == 1) & (sequence == 0) & positive).sum()),
        "sequence_only_tp": int(((mulcas == 0) & (sequence == 1) & positive).sum()),
        "fn_by_both": int(((mulcas == 0) & (sequence == 0) & positive).sum()),
        "theoretical_union_recall": union_tp / int(positive.sum()) if positive.sum() else None,
        "oracle_label": "non_deployable_union_diagnostic",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    results = run_phase8r5c1(args.config)
    print(json.dumps({"status": results["status"], "eligibility": results.get("eligibility")}, indent=2))


if __name__ == "__main__":
    main()
