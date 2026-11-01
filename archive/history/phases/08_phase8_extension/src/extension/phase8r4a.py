"""Creation-code feasibility audit and frozen deployment-time feature corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import yaml

from src.extension.creation_code import RuntimeBoundary, analyze_runtime_boundary
from src.extension.creation_features import (
    BOUNDARY_FEATURE_NAMES,
    COMBINED_FEATURE_NAMES,
    CREATION_FULL_FEATURE_NAMES,
    DEPLOYMENT_INIT_FEATURE_NAMES,
    boundary_feature_vector,
    extract_creation_full_features,
    extract_init_features,
)
from src.extension.phase8r2a import write_deterministic_npz
from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/extension/phase8r4a.yaml"
PERIODS = tuple(f"P{index}" for index in range(6))
EXPECTED_PERIOD_ROWS = {"P0": 104, "P1": 305, "P2": 5141, "P3": 289, "P4": 150, "P5": 509}
SUMMARY_FEATURES = (
    "creation_byte_length", "creation_instruction_count",
    "creation_unique_opcode_count", "creation_opcode_entropy",
    "creation_runtime_length_ratio", "init_region_available",
    "init_byte_length", "init_instruction_count", "init_basic_block_count",
    "trailing_byte_length",
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset", "period_definitions", "output_dir", "artifact_path"):
        config[key] = _resolve(config[key])
    config["frozen_manifests"] = [_resolve(item) for item in config["frozen_manifests"]]
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["sources"]["creation_bytecode"] != "creationCode":
        raise ValueError("creationCode must be the authoritative deployment source.")
    if config["sources"]["runtime_bytecode"] != "contractCode":
        raise ValueError("contractCode may only define the runtime boundary.")
    execution = config["execution"]
    if execution["classifier_training"] or execution["model_evaluation"] or execution["p5_model_evaluation"]:
        raise ValueError("Phase 8R4A cannot train or evaluate a model.")
    if config["boundary_policy"]["init_available_only_when"] != "exactly_one_nonempty_runtime_occurrence":
        raise ValueError("The conservative init-boundary policy changed.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in config["frozen_manifests"]:
        observed.update(verify_frozen_manifest(manifest))
    return observed


def _period_for_block(block: int, definitions: list[dict[str, Any]]) -> str:
    for item in definitions:
        if int(item["block_min"]) <= block <= int(item["block_max"]):
            return str(item["period"])
    raise ValueError(f"Block {block} is outside frozen Phase 7 period boundaries.")


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(
        config["dataset"],
        usecols=["address", "createdBlockNumber", "label", "creationCode", "contractCode"],
        low_memory=False,
    ).reset_index(names="row_index")
    if len(frame) != 6498 or frame["row_index"].duplicated().any():
        raise AssertionError("Expected exactly 6,498 uniquely indexed contracts.")
    definitions = json.loads(config["period_definitions"].read_text(encoding="utf-8"))
    frame["period"] = [
        _period_for_block(int(block), definitions) for block in frame["createdBlockNumber"]
    ]
    if frame.groupby("period").size().to_dict() != EXPECTED_PERIOD_ROWS:
        raise AssertionError("Frozen P0-P5 membership changed.")
    return frame


def _extract_one(
    creation_code: object, runtime_code: object, small_constant_max: int
) -> tuple[RuntimeBoundary, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    boundary = analyze_runtime_boundary(creation_code, runtime_code)
    runtime_length = len(boundary.runtime) if boundary.runtime is not None else 0
    full, creation_cfg = extract_creation_full_features(
        creation_code, runtime_length, small_constant_max=small_constant_max
    )
    init, init_cfg = extract_init_features(
        boundary.init_region, runtime_length, small_constant_max=small_constant_max
    )
    diagnostic = boundary_feature_vector(boundary)
    combined = np.concatenate([full, init, diagnostic]).astype(np.float32)
    if combined.shape != (len(COMBINED_FEATURE_NAMES),) or not np.isfinite(combined).all():
        raise AssertionError("Combined creation feature schema is invalid.")
    record = {
        "creation_byte_length": len(boundary.creation) if boundary.creation is not None else 0,
        "runtime_byte_length": runtime_length,
        "creation_runtime_length_ratio": (
            len(boundary.creation) / runtime_length
            if boundary.creation is not None and runtime_length else 0.0
        ),
        "creation_parseable": boundary.creation_parseable,
        "runtime_parseable": boundary.runtime_parseable,
        "creation_instruction_count": len(creation_cfg.instructions),
        "creation_unique_opcode_count": len({item.name for item in creation_cfg.instructions}),
        "creation_opcode_entropy": float(full[CREATION_FULL_FEATURE_NAMES.index("creation_opcode_entropy")]),
        "malformed_truncated_push_count": creation_cfg.truncated_push_count,
        "unsupported_opcode_count": creation_cfg.unsupported_opcode_count,
        "runtime_exact_substring_found": boundary.runtime_exact_substring_found,
        "runtime_occurrence_count": boundary.runtime_occurrence_count,
        "first_runtime_substring_byte_offset": boundary.first_runtime_substring_byte_offset,
        "last_runtime_substring_byte_offset": boundary.last_runtime_substring_byte_offset,
        "bytes_before_first_runtime_match": boundary.bytes_before_first_runtime_match,
        "bytes_after_last_runtime_match": boundary.bytes_after_last_runtime_match,
        "runtime_fraction_of_creation": boundary.runtime_fraction_of_creation,
        "boundary_status": boundary.boundary_status,
        "init_region_available": boundary.init_region_available,
        "init_byte_length": len(boundary.init_region) if boundary.init_region is not None else 0,
        "init_instruction_count": len(init_cfg.instructions) if init_cfg is not None else 0,
        "init_basic_block_count": len(init_cfg.blocks) if init_cfg is not None else 0,
        "trailing_byte_length": len(boundary.trailing_region) if boundary.trailing_region is not None else 0,
        "trailing_nonempty": bool(boundary.trailing_region),
        "trailing_runtime_length_ratio": (
            len(boundary.trailing_region) / runtime_length
            if boundary.trailing_region is not None and runtime_length else 0.0
        ),
    }
    return boundary, full, init, combined, record


def performance_preflight(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    sample_size = int(config["preflight"]["sample_size"])
    sample = frame.sort_values(
        ["createdBlockNumber", "row_index"], kind="mergesort"
    ).head(sample_size)
    small = int(config["features"]["small_constant_max_inclusive"])
    started = time.perf_counter()
    for row in sample.itertuples(index=False):
        _extract_one(row.creationCode, row.contractCode, small)
    elapsed = time.perf_counter() - started
    estimate = elapsed / sample_size * len(frame)
    matrix_bytes = len(frame) * (
        len(CREATION_FULL_FEATURE_NAMES)
        + len(DEPLOYMENT_INIT_FEATURE_NAMES)
        + len(COMBINED_FEATURE_NAMES)
    ) * np.dtype(np.float32).itemsize + len(frame) * np.dtype(np.int64).itemsize
    return {
        "cpu_model": platform.processor(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "system_ram_bytes": int(psutil.virtual_memory().total),
        "execution_device": "cpu",
        "sample_policy": config["preflight"]["sample_policy"],
        "sample_size": sample_size,
        "sample_runtime_seconds": elapsed,
        "mean_milliseconds_per_contract": elapsed / sample_size * 1000.0,
        "estimated_full_corpus_runtime_seconds": estimate,
        "estimated_uncompressed_matrix_size_bytes": int(matrix_bytes),
        "stop_limit_seconds": float(config["preflight"]["maximum_estimated_runtime_seconds"]),
        "full_processing_authorized": estimate <= float(config["preflight"]["maximum_estimated_runtime_seconds"]),
    }


def construct_features(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    full_rows: list[np.ndarray] = []
    init_rows: list[np.ndarray] = []
    combined_rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    small = int(config["features"]["small_constant_max_inclusive"])
    # Labels are deliberately absent from the extraction call.
    for row in frame.itertuples(index=False):
        _, full, init, combined, record = _extract_one(
            row.creationCode, row.contractCode, small
        )
        full_rows.append(full)
        init_rows.append(init)
        combined_rows.append(combined)
        records.append({
            "row_index": int(row.row_index), "address": str(row.address),
            "createdBlockNumber": int(row.createdBlockNumber), "period": str(row.period),
            **record,
        })
    full_matrix = np.vstack(full_rows).astype(np.float32)
    init_matrix = np.vstack(init_rows).astype(np.float32)
    combined_matrix = np.vstack(combined_rows).astype(np.float32)
    for matrix in (full_matrix, init_matrix, combined_matrix):
        if not np.isfinite(matrix).all():
            raise AssertionError("Every generated feature must be finite.")
    return full_matrix, init_matrix, combined_matrix, pd.DataFrame(records)


def _distribution(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    return {
        "min": float(numeric.min()), "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()), "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)), "q75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)), "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)), "max": float(numeric.max()),
    }


def creation_audit(records: pd.DataFrame) -> dict[str, Any]:
    status = records["boundary_status"].value_counts().to_dict()
    return {
        "contracts_processed": len(records),
        "creation_parseable_contracts": int(records["creation_parseable"].sum()),
        "runtime_parseable_contracts": int(records["runtime_parseable"].sum()),
        "contracts_with_truncated_push": int(records["malformed_truncated_push_count"].gt(0).sum()),
        "malformed_truncated_push_count": int(records["malformed_truncated_push_count"].sum()),
        "contracts_with_unsupported_opcodes": int(records["unsupported_opcode_count"].gt(0).sum()),
        "unsupported_opcode_count": int(records["unsupported_opcode_count"].sum()),
        "runtime_exact_substring_found_count": int(records["runtime_exact_substring_found"].sum()),
        "runtime_unique_match_count": int(status.get("unique", 0)),
        "runtime_ambiguous_match_count": int(status.get("ambiguous", 0)),
        "runtime_not_found_count": int(status.get("not_found", 0)),
        "invalid_creation_count": int(status.get("invalid_creation", 0)),
        "invalid_runtime_count": int(status.get("invalid_runtime", 0)),
        "empty_runtime_count": int(status.get("empty_runtime", 0)),
        "init_region_available_count": int(records["init_region_available"].sum()),
        "trailing_nonempty_count": int(records["trailing_nonempty"].sum()),
        "creation_byte_length": _distribution(records["creation_byte_length"]),
        "runtime_byte_length": _distribution(records["runtime_byte_length"]),
        "creation_runtime_length_ratio": _distribution(records["creation_runtime_length_ratio"]),
        "init_byte_length": _distribution(records["init_byte_length"]),
        "trailing_byte_length": _distribution(records["trailing_byte_length"]),
    }


def period_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        selected = records[records["period"].eq(period)]
        row: dict[str, Any] = {
            "period": period, "contract_count": len(selected),
            "runtime_match_rate": float(selected["runtime_exact_substring_found"].mean()),
            "unique_runtime_match_rate": float(selected["boundary_status"].eq("unique").mean()),
            "ambiguous_runtime_match_rate": float(selected["boundary_status"].eq("ambiguous").mean()),
            "init_region_availability_rate": float(selected["init_region_available"].mean()),
            "trailing_nonempty_rate": float(selected["trailing_nonempty"].mean()),
        }
        for name in (
            "creation_byte_length", "runtime_byte_length", "creation_runtime_length_ratio",
            "creation_instruction_count", "init_byte_length", "init_basic_block_count",
            "trailing_byte_length",
        ):
            distribution = _distribution(selected[name])
            for statistic in ("median", "mean", "std", "q25", "q75"):
                row[f"{name}_{statistic}"] = distribution[statistic]
        rows.append(row)
    return pd.DataFrame(rows)


def label_summary(records: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    joined = records.merge(labels[["row_index", "label"]], on="row_index", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for label in (0, 1):
        selected = joined[joined["label"].eq(label)]
        for feature in SUMMARY_FEATURES:
            values = selected[feature].astype(float)
            rows.append({
                "label": label, "class_name": "ponzi" if label else "benign",
                "feature": feature, "count": len(values),
                "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                "q25": float(values.quantile(0.25)), "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "descriptive_only": True,
            })
    return pd.DataFrame(rows)


def boundary_statistics(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(records)
    for status in ("unique", "ambiguous", "not_found", "invalid_creation", "invalid_runtime", "empty_runtime"):
        selected = records[records["boundary_status"].eq(status)]
        rows.append({
            "boundary_status": status, "contract_count": len(selected),
            "fraction_of_corpus": len(selected) / total,
            "mean_runtime_occurrences": float(selected["runtime_occurrence_count"].mean()) if len(selected) else 0.0,
            "median_bytes_before_first_match": float(selected["bytes_before_first_runtime_match"].median()) if selected["bytes_before_first_runtime_match"].notna().any() else None,
            "median_bytes_after_last_match": float(selected["bytes_after_last_runtime_match"].median()) if selected["bytes_after_last_runtime_match"].notna().any() else None,
            "init_region_available_count": int(selected["init_region_available"].sum()),
        })
    return pd.DataFrame(rows)


def feature_schema(config: dict[str, Any]) -> dict[str, Any]:
    schema = {
        "schema_version": config["features"]["schema_version"],
        "dtype": "float32", "raw_constants_emitted": False,
        "views": {
            "creation_full": {"dimension": len(CREATION_FULL_FEATURE_NAMES), "ordered_feature_names": list(CREATION_FULL_FEATURE_NAMES)},
            "deployment_init": {"dimension": len(DEPLOYMENT_INIT_FEATURE_NAMES), "ordered_feature_names": list(DEPLOYMENT_INIT_FEATURE_NAMES)},
            "creation_full_plus_init_and_boundary": {"dimension": len(COMBINED_FEATURE_NAMES), "ordered_feature_names": list(COMBINED_FEATURE_NAMES)},
        },
        "boundary_feature_names": list(BOUNDARY_FEATURE_NAMES),
        "unavailable_init_policy": "availability=0 and all deployment_init features=0",
        "boundary_policy": config["boundary_policy"],
    }
    schema["schema_sha256"] = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return schema


def build_report(
    audit: dict[str, Any], periods: pd.DataFrame, schema: dict[str, Any],
    preflight: dict[str, Any], artifact: dict[str, Any], runtime_seconds: float,
) -> str:
    period_rows = [
        f"| {row.period} | {int(row.contract_count)} | {row.runtime_match_rate:.4f} | "
        f"{row.init_region_availability_rate:.4f} | {row.creation_runtime_length_ratio_median:.4f} | "
        f"{row.init_byte_length_median:.1f} | {row.trailing_nonempty_rate:.4f} |"
        for row in periods.itertuples(index=False)
    ]
    total = audit["contracts_processed"]
    return "\n".join([
        "# Phase 8R4A: Creation-Code Feasibility And Features", "",
        "This phase constructs and audits deployment-time features only. No classifier was trained, no model metric was calculated, and P5 was used only in descriptive period summaries.", "",
        "## Runtime Boundary Policy", "",
        "The normalized deployed runtime must occur exactly once as a contiguous byte substring of `creationCode`. Only then is the prefix treated as the deployment-init region. Multiple or missing matches never receive an inferred init boundary. Bytes after the last exact runtime match are retained as trailing-data diagnostics and are not interpreted as constructor arguments.", "",
        "## Preflight", "",
        f"The deterministic 100-contract CPU sample took {preflight['sample_runtime_seconds']:.3f}s ({preflight['mean_milliseconds_per_contract']:.3f} ms/contract), projecting {preflight['estimated_full_corpus_runtime_seconds']:.1f}s for the corpus. Actual processing took {runtime_seconds:.1f}s.", "",
        "## Corpus Audit", "",
        f"- Parseable creationCode: {audit['creation_parseable_contracts']:,}/{total:,}",
        f"- Exact runtime substring found: {audit['runtime_exact_substring_found_count']:,}/{total:,}",
        f"- Unique / ambiguous / absent runtime matches: {audit['runtime_unique_match_count']:,} / {audit['runtime_ambiguous_match_count']:,} / {audit['runtime_not_found_count']:,}",
        f"- Init region available: {audit['init_region_available_count']:,}/{total:,}",
        f"- Nonempty trailing region: {audit['trailing_nonempty_count']:,}/{total:,}", "",
        "## Frozen Views", "",
        *(f"- `{name}`: {item['dimension']} features" for name, item in schema["views"].items()),
        f"- Matrix SHA-256: `{artifact['sha256']}`",
        f"- Compressed size: {artifact['size_bytes']:,} bytes", "",
        "## Period Description", "", "| Period | Rows | Runtime match | Init available | Median creation/runtime ratio | Median init bytes | Trailing nonempty |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |", *period_rows, "",
        "These period and class summaries are descriptive only. They did not alter the frozen feature schema. H8R4 remains unevaluated until a later modeling phase.", "",
    ])


def run(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    frozen = verify_frozen_inputs(config)
    frame = load_dataset(config)
    preflight = performance_preflight(frame, config)
    if not preflight["full_processing_authorized"]:
        config["output_dir"].mkdir(parents=True, exist_ok=True)
        (config["output_dir"] / "runtime_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        raise RuntimeError("Projected Phase 8R4A runtime exceeds 30 minutes.")
    full, init, combined, records = construct_features(frame, config)
    schema = feature_schema(config)
    arrays = {
        "row_index": frame["row_index"].to_numpy(dtype=np.int64),
        "creation_full": full,
        "deployment_init": init,
        "creation_full_plus_init_and_boundary": combined,
    }
    write_deterministic_npz(config["artifact_path"], arrays)
    first_hash = sha256_file(config["artifact_path"])
    write_deterministic_npz(config["artifact_path"], arrays)
    second_hash = sha256_file(config["artifact_path"])
    if first_hash != second_hash:
        raise AssertionError("Repeated matrix serialization was not deterministic.")
    audit = creation_audit(records)
    periods = period_summary(records)
    labels = frame[["row_index", "label"]].copy()
    labels["label"] = labels["label"].astype(int)
    label_descriptive = label_summary(records, labels)
    boundaries = boundary_statistics(records)
    actual_runtime = time.perf_counter() - started
    preflight["actual_full_run_seconds"] = actual_runtime
    preflight["actual_output_size_bytes"] = config["artifact_path"].stat().st_size
    artifact = {
        "path": config["artifact_path"].resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": second_hash, "size_bytes": config["artifact_path"].stat().st_size,
        "row_count": len(frame),
        "view_dimensions": {"creation_full": full.shape[1], "deployment_init": init.shape[1], "creation_full_plus_init_and_boundary": combined.shape[1]},
        "labels_stored": False,
    }
    results = {
        "phase": "8R4A", "construction_only": True,
        "classifier_trained": False, "model_evaluation_performed": False,
        "p5_model_evaluation_performed": False,
        "p5_usage": "descriptive_period_summary_only",
        "h8r4_evaluated": False,
        "dataset_sha256": sha256_file(config["dataset"]),
        "config_sha256": config["config_sha256"],
        "frozen_artifact_hashes": frozen,
        "preflight": preflight, "creation_audit": audit,
        "feature_schema_sha256": schema["schema_sha256"],
        "feature_artifact": artifact,
        "determinism": {"first_hash": first_hash, "second_hash": second_hash, "hashes_equal": first_hash == second_hash},
        "labels_used_during_feature_extraction": False,
        "labels_joined_after_schema_freeze_for_description_only": True,
    }
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase8r4a_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "creation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "creation_feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    records.to_csv(output / "creation_feature_summary.csv", index=False)
    periods.to_csv(output / "period_creation_summary.csv", index=False)
    label_descriptive.to_csv(output / "label_descriptive_summary.csv", index=False)
    boundaries.to_csv(output / "runtime_boundary_statistics.csv", index=False)
    (output / "runtime_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    (output / "phase8r4a_report.md").write_text(
        build_report(audit, periods, schema, preflight, artifact, actual_runtime), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    results = run(load_config(args.config))
    print(json.dumps({
        "contracts": results["creation_audit"]["contracts_processed"],
        "init_available": results["creation_audit"]["init_region_available_count"],
        "feature_matrix_sha256": results["feature_artifact"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
