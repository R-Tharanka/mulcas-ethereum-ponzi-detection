"""Build and validate the frozen Phase 8R2A static EVM CFG corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.extension.cfg import EDGE_TYPES, TERMINATORS, ContractCFG, build_cfg
from src.extension.graph_features import (
    FEATURE_DIMENSION,
    FEATURE_NAMES,
    TERMINATOR_FEATURES,
    graph_arrays,
)
from src.features.operands import OPCODE_NAMES
from src.features.opcode_categories import CATEGORY_NAMES, OPCODE_CATEGORIES
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "extension" / "phase8r2a.yaml"
TERMINATOR_CODES = {name: index for index, name in enumerate(TERMINATOR_FEATURES)}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset", "phase7_period_definitions", "frozen_phase7_manifest",
        "frozen_phase8r0_manifest", "frozen_phase8r1a_manifest", "output_dir",
        "artifact_dir",
    ):
        config[key] = _resolve(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    execution = config["execution"]
    if any(execution[key] for key in ("train_classifier", "train_gnn", "evaluate_model", "evaluate_p5_model_performance")):
        raise ValueError("Phase 8R2A cannot train or evaluate a model.")
    if config["source"]["bytecode_column"] != "contractCode":
        raise ValueError("contractCode must remain the authoritative graph source.")
    if config["jump_resolution"]["unresolved_fanout"]:
        raise ValueError("Unresolved jumps must not receive fabricated edges.")
    return config


def verify_frozen_chain(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in (
        config["frozen_phase7_manifest"],
        config["frozen_phase8r0_manifest"],
        config["frozen_phase8r1a_manifest"],
    ):
        for relative, digest in verify_frozen_manifest(manifest).items():
            if relative in observed and observed[relative] != digest:
                raise RuntimeError(f"Conflicting frozen hash for {relative}")
            observed[relative] = digest
    return observed


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write byte-stable NPZ data using fixed ZIP metadata and sorted keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def graph_content_hash(cfg: ContractCFG) -> str:
    x, edge_index, edge_type = graph_arrays(cfg)
    digest = hashlib.sha256()
    for array in (x, edge_index, edge_type):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _terminator_code(name: str) -> int:
    lowered = name.lower()
    return TERMINATOR_CODES.get(lowered, TERMINATOR_CODES["other_terminal"])


def chunk_arrays(graphs: list[tuple[int, ContractCFG]]) -> dict[str, np.ndarray]:
    """Pack multiple graphs with pointer arrays for later framework conversion."""
    x_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    edge_type_parts: list[np.ndarray] = []
    instruction_pc: list[int] = []
    instruction_opcode: list[int] = []
    instruction_push_width: list[int] = []
    instruction_size: list[int] = []
    instruction_flags: list[int] = []
    operand_data: list[int] = []
    operand_ptr = [0]
    block_start_pc: list[int] = []
    block_end_pc: list[int] = []
    block_instruction_start: list[int] = []
    block_instruction_count: list[int] = []
    block_byte_length: list[int] = []
    block_terminator: list[int] = []
    node_ptr = [0]
    edge_ptr = [0]
    graph_instruction_ptr = [0]
    node_offset = 0
    instruction_offset = 0

    for _, cfg in graphs:
        x, edge_index, edge_type = graph_arrays(cfg)
        x_parts.append(x)
        edge_parts.append(edge_index + node_offset)
        edge_type_parts.append(edge_type)
        local_instruction = 0
        for instruction in cfg.instructions:
            instruction_pc.append(instruction.pc)
            instruction_opcode.append(instruction.opcode)
            instruction_push_width.append(instruction.push_width)
            instruction_size.append(instruction.size)
            instruction_flags.append(int(instruction.truncated) | (int(instruction.supported) << 1))
            operand_data.extend(instruction.operand)
            operand_ptr.append(len(operand_data))
        for block in cfg.blocks:
            block_start_pc.append(block.start_pc)
            block_end_pc.append(block.end_pc)
            block_instruction_start.append(instruction_offset + local_instruction)
            block_instruction_count.append(block.instruction_count)
            block_byte_length.append(block.byte_length)
            block_terminator.append(_terminator_code(block.terminator_opcode))
            local_instruction += block.instruction_count
        node_offset += len(cfg.blocks)
        instruction_offset += len(cfg.instructions)
        node_ptr.append(node_offset)
        edge_ptr.append(edge_ptr[-1] + len(cfg.edges))
        graph_instruction_ptr.append(instruction_offset)

    return {
        "x": np.vstack(x_parts).astype(np.float32) if x_parts else np.empty((0, FEATURE_DIMENSION), dtype=np.float32),
        "edge_index": np.hstack(edge_parts).astype(np.int64) if edge_parts else np.empty((2, 0), dtype=np.int64),
        "edge_type": np.concatenate(edge_type_parts).astype(np.int8) if edge_type_parts else np.empty(0, dtype=np.int8),
        "row_index": np.asarray([row for row, _ in graphs], dtype=np.int64),
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "graph_instruction_ptr": np.asarray(graph_instruction_ptr, dtype=np.int64),
        "instruction_pc": np.asarray(instruction_pc, dtype=np.int32),
        "instruction_opcode": np.asarray(instruction_opcode, dtype=np.uint8),
        "instruction_push_width": np.asarray(instruction_push_width, dtype=np.uint8),
        "instruction_size": np.asarray(instruction_size, dtype=np.uint8),
        "instruction_flags": np.asarray(instruction_flags, dtype=np.uint8),
        "operand_data": np.asarray(operand_data, dtype=np.uint8),
        "operand_ptr": np.asarray(operand_ptr, dtype=np.int64),
        "block_start_pc": np.asarray(block_start_pc, dtype=np.int32),
        "block_end_pc": np.asarray(block_end_pc, dtype=np.int32),
        "block_instruction_start": np.asarray(block_instruction_start, dtype=np.int64),
        "block_instruction_count": np.asarray(block_instruction_count, dtype=np.int32),
        "block_byte_length": np.asarray(block_byte_length, dtype=np.int32),
        "block_terminator": np.asarray(block_terminator, dtype=np.int8),
    }


def graph_structure_statistics(cfg: ContractCFG) -> dict[str, Any]:
    nodes = len(cfg.blocks)
    edges = len(cfg.edges)
    indegree = np.zeros(nodes, dtype=int)
    outdegree = np.zeros(nodes, dtype=int)
    parent = list(range(nodes))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in cfg.edges:
        outdegree[edge.source] += 1
        indegree[edge.target] += 1
        union(edge.source, edge.target)
    components = len({find(index) for index in range(nodes)}) if nodes else 0
    type_counts = Counter(edge.edge_type for edge in cfg.edges)
    x, _, _ = graph_arrays(cfg)
    return {
        "node_count": nodes,
        "edge_count": edges,
        "fall_through_edge_count": type_counts[EDGE_TYPES["FALL_THROUGH"]],
        "static_jump_edge_count": type_counts[EDGE_TYPES["STATIC_JUMP"]],
        "conditional_jump_edge_count": type_counts[EDGE_TYPES["CONDITIONAL_JUMP"]],
        "unresolved_jump_count": cfg.jump_statistics.unresolved,
        "invalid_jump_target_count": cfg.jump_statistics.invalid_target,
        "truncated_push_count": cfg.truncated_push_count,
        "unsupported_opcode_count": cfg.unsupported_opcode_count,
        "connected_component_count": components,
        "zero_outdegree_node_count": int((outdegree == 0).sum()),
        "graph_density": float(edges / (nodes * (nodes - 1))) if nodes > 1 else 0.0,
        "max_in_degree": int(indegree.max()) if nodes else 0,
        "max_out_degree": int(outdegree.max()) if nodes else 0,
        "non_finite_feature_count": int((~np.isfinite(x)).sum()),
        "invalid_hex": cfg.invalid_hex,
        "empty_bytecode": cfg.empty,
    }


def _distribution(values: Iterable[int | float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {key: 0.0 for key in ("min", "median", "mean", "p90", "p95", "p99", "max")}
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def _period_for_block(block: int, definitions: list[dict[str, Any]]) -> str:
    for item in definitions:
        if int(item["block_min"]) <= block <= int(item["block_max"]):
            return str(item["period"])
    raise ValueError(f"Block {block} is outside frozen Phase 7 period boundaries.")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def graph_schema(config: dict[str, Any]) -> dict[str, Any]:
    opcode_names = {
        f"0x{opcode:02x}": OPCODE_NAMES.get(opcode, f"UNKNOWN_0X{opcode:02X}")
        for opcode in range(256)
    }
    return {
        "schema_version": config["schema"]["version"],
        "authoritative_source": "contractCode",
        "node_feature_dimension": FEATURE_DIMENSION,
        "ordered_node_feature_names": list(FEATURE_NAMES),
        "canonical_category_order": list(CATEGORY_NAMES),
        "canonical_opcode_categories": {
            name: sorted(values) for name, values in OPCODE_CATEGORIES.items()
        },
        "opcode_byte_to_canonical_name": opcode_names,
        "edge_types": EDGE_TYPES,
        "terminator_codes": TERMINATOR_CODES,
        "instruction_flags": {"bit_0": "truncated_push", "bit_1": "supported_opcode"},
        "basic_block_policy": config["basic_blocks"],
        "jump_resolution_policy": config["jump_resolution"],
        "node_feature_exclusions": [
            "label", "contract_address", "creator", "raw_push_constant_identity",
            "function_selector_identity", "embedded_address_identity",
        ],
        "chunk_arrays": {
            "x": "concatenated float32 node features",
            "edge_index": "int64 [2, edges], node indices local to chunk concatenation",
            "edge_type": "int8 edge type codes",
            "row_index": "int64 graph identifiers",
            "node_ptr": "int64 graph-to-node boundaries",
            "edge_ptr": "int64 graph-to-edge boundaries",
            "graph_instruction_ptr": "int64 graph-to-instruction boundaries",
            "instruction_pc": "int32 instruction byte offsets",
            "instruction_opcode": "uint8 opcode bytes; names derive from schema table",
            "instruction_push_width": "uint8 declared PUSH widths",
            "instruction_size": "uint8 actual decoded instruction sizes",
            "instruction_flags": "uint8 truncation/support bits",
            "operand_data": "uint8 validated operand bytes, provenance only",
            "operand_ptr": "int64 instruction-to-operand boundaries",
            "block_start_pc": "int32 block start PCs",
            "block_end_pc": "int32 inclusive block end PCs",
            "block_instruction_start": "int64 instruction-array offset per block",
            "block_instruction_count": "int32 instructions per block",
            "block_byte_length": "int32 actual decoded bytes per block",
            "block_terminator": "int8 terminator category code",
        },
    }


def _aggregate_graphs(frame: pd.DataFrame) -> dict[str, Any]:
    total_jumps = int(frame["jump_total"].sum())
    resolved = int(frame["jump_resolved"].sum())
    return {
        "contracts_processed": int(len(frame)),
        "graphs_created": int(len(frame)),
        "zero_node_graphs": int(frame["node_count"].eq(0).sum()),
        "zero_edge_graphs": int(frame["edge_count"].eq(0).sum()),
        "node_count": _distribution(frame["node_count"]),
        "edge_count": _distribution(frame["edge_count"]),
        "unresolved_jump_count": _distribution(frame["unresolved_jump_count"]),
        "truncated_push_count": _distribution(frame["truncated_push_count"]),
        "jump_total": total_jumps,
        "jump_resolved": resolved,
        "jump_unresolved": int(frame["jump_unresolved"].sum()),
        "jump_invalid_target": int(frame["jump_invalid_target"].sum()),
        "static_resolution_rate": float(resolved / total_jumps) if total_jumps else 0.0,
        "graph_feature_dimension": FEATURE_DIMENSION,
        "non_finite_feature_count": int(frame["non_finite_feature_count"].sum()),
        "invalid_hex_contracts": int(frame["invalid_hex"].sum()),
        "empty_bytecode_contracts": int(frame["empty_bytecode"].sum()),
        "contracts_with_truncated_push": int(frame["truncated_push_count"].gt(0).sum()),
        "contracts_with_unresolved_jump": int(frame["unresolved_jump_count"].gt(0).sum()),
    }


def _period_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in ("P0", "P1", "P2", "P3", "P4", "P5"):
        selected = frame[frame["period"] == period]
        aggregate = _aggregate_graphs(selected)
        row: dict[str, Any] = {
            "period": period,
            "contract_count": aggregate["contracts_processed"],
            "zero_node_graphs": aggregate["zero_node_graphs"],
            "zero_edge_graphs": aggregate["zero_edge_graphs"],
            "jump_total": aggregate["jump_total"],
            "jump_resolved": aggregate["jump_resolved"],
            "jump_unresolved": aggregate["jump_unresolved"],
            "jump_invalid_target": aggregate["jump_invalid_target"],
            "static_resolution_rate": aggregate["static_resolution_rate"],
            "non_finite_feature_count": aggregate["non_finite_feature_count"],
        }
        for measure in ("node_count", "edge_count", "unresolved_jump_count", "truncated_push_count"):
            for statistic, value in aggregate[measure].items():
                row[f"{measure}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_graph_corpus(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Stream all contracts into deterministic ignored graph chunks and audits."""
    frozen_hashes = verify_frozen_chain(config)
    definitions = json.loads(config["phase7_period_definitions"].read_text(encoding="utf-8"))
    schema = graph_schema(config)
    schema_hash = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    chunk_size = int(config["schema"]["chunk_size"])
    records: list[dict[str, Any]] = []
    jump_records: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    global_row = 0
    first_graph_hashes: list[str] = []

    reader = pd.read_csv(
        config["dataset"],
        usecols=["address", "contractCode", "createdBlockNumber"],
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk_id, chunk in enumerate(reader):
        packed: list[tuple[int, ContractCFG]] = []
        chunk_nodes = chunk_edges = 0
        first_row = global_row
        for row in chunk.itertuples(index=False):
            row_index = global_row
            global_row += 1
            cfg = build_cfg(row.contractCode)
            if row_index == 0:
                first_graph_hashes = [graph_content_hash(cfg), graph_content_hash(build_cfg(row.contractCode))]
            structural = graph_structure_statistics(cfg)
            block_number = int(row.createdBlockNumber)
            period = _period_for_block(block_number, definitions)
            record = {
                "row_index": row_index,
                "address": str(row.address),
                "createdBlockNumber": block_number,
                "period": period,
                **structural,
                "instruction_count": len(cfg.instructions),
                "jump_total": cfg.jump_statistics.total,
                "jump_resolved": cfg.jump_statistics.resolved,
                "jump_unresolved": cfg.jump_statistics.unresolved,
                "jump_invalid_target": cfg.jump_statistics.invalid_target,
                "jump_resolution_rate": cfg.jump_statistics.resolution_rate,
            }
            records.append(record)
            jump_records.append({
                "row_index": row_index,
                "period": period,
                "total_jump_jumpi": cfg.jump_statistics.total,
                "statically_resolved": cfg.jump_statistics.resolved,
                "unresolved": cfg.jump_statistics.unresolved,
                "invalid_target": cfg.jump_statistics.invalid_target,
                "resolution_rate": cfg.jump_statistics.resolution_rate,
            })
            packed.append((row_index, cfg))
            chunk_nodes += len(cfg.blocks)
            chunk_edges += len(cfg.edges)

        arrays = chunk_arrays(packed)
        artifact = config["artifact_dir"] / f"graphs_{chunk_id:04d}.npz"
        write_deterministic_npz(artifact, arrays)
        artifacts.append({
            "chunk_id": chunk_id,
            "path": _relative(artifact),
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
            "graph_count": len(packed),
            "first_row_index": first_row,
            "last_row_index": global_row - 1,
            "node_count": chunk_nodes,
            "edge_count": chunk_edges,
        })

    graph_frame = pd.DataFrame(records)
    jump_frame = pd.DataFrame(jump_records)
    if len(graph_frame) != 6498 or graph_frame["row_index"].nunique() != 6498:
        raise AssertionError("Every one of the 6,498 contracts must receive exactly one graph.")
    aggregate = _aggregate_graphs(graph_frame)
    if aggregate["graphs_created"] != 6498 or aggregate["non_finite_feature_count"]:
        raise AssertionError("Graph corpus is incomplete or contains non-finite features.")
    if first_graph_hashes[0] != first_graph_hashes[1]:
        raise AssertionError("Repeated graph construction was not deterministic.")

    corpus_digest = hashlib.sha256()
    for artifact in artifacts:
        corpus_digest.update(f"{artifact['path']}:{artifact['sha256']}\n".encode("utf-8"))
    manifest = {
        "schema_version": config["schema"]["version"],
        "schema_sha256": schema_hash,
        "source_dataset": _relative(config["dataset"]),
        "source_dataset_sha256": sha256_file(config["dataset"]),
        "authoritative_bytecode_field": "contractCode",
        "chunk_size": chunk_size,
        "graph_count": len(graph_frame),
        "node_feature_dimension": FEATURE_DIMENSION,
        "total_nodes": int(graph_frame["node_count"].sum()),
        "total_edges": int(graph_frame["edge_count"].sum()),
        "artifact_count": len(artifacts),
        "artifact_size_bytes": int(sum(item["size_bytes"] for item in artifacts)),
        "corpus_sha256": corpus_digest.hexdigest(),
        "artifacts": artifacts,
        "labels_in_graph_artifacts": False,
        "torch_required": False,
    }
    results = {
        "phase": "8R2A",
        "hypothesis": config["hypothesis"],
        "construction_only": True,
        "classifier_trained": False,
        "gnn_trained": False,
        "model_evaluation_performed": False,
        "p5_model_evaluation_performed": False,
        "labels_loaded_for_graph_construction": False,
        "dataset_sha256": sha256_file(config["dataset"]),
        "config_sha256": config["config_sha256"],
        "frozen_artifact_hashes": frozen_hashes,
        "aggregate_graph_audit": aggregate,
        "graph_manifest_summary": {key: value for key, value in manifest.items() if key != "artifacts"},
        "determinism": {
            "repeated_first_graph_hashes": first_graph_hashes,
            "repeated_graph_hashes_equal": first_graph_hashes[0] == first_graph_hashes[1],
            "deterministic_npz_writer": True,
        },
        "frozen_policies": {
            "graph_schema_version": config["schema"]["version"],
            "node_feature_schema_sha256": schema_hash,
            "edge_types": EDGE_TYPES,
            "jump_resolution": config["jump_resolution"],
            "basic_blocks": config["basic_blocks"],
            "artifact_schema": schema["chunk_arrays"],
        },
    }
    return results, manifest, schema, graph_frame, _period_statistics(graph_frame), jump_frame


def build_report(
    results: dict[str, Any], manifest: dict[str, Any], periods: pd.DataFrame
) -> str:
    audit = results["aggregate_graph_audit"]
    distribution_rows = [
        "| Measure | Min | Median | Mean | P90 | P95 | P99 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("Nodes", "node_count"),
        ("Edges", "edge_count"),
        ("Unresolved jumps", "unresolved_jump_count"),
        ("Truncated PUSH", "truncated_push_count"),
    ):
        item = audit[key]
        distribution_rows.append(
            f"| {label} | {item['min']} | {item['median']:.2f} | {item['mean']:.2f} | "
            f"{item['p90']:.2f} | {item['p95']:.2f} | {item['p99']:.2f} | {item['max']} |"
        )
    period_rows = [
        "| Period | Graphs | Median nodes | Mean nodes | Median edges | Mean edges | Resolution rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in periods.itertuples(index=False):
        period_rows.append(
            f"| {row.period} | {row.contract_count} | {row.node_count_median:.2f} | "
            f"{row.node_count_mean:.2f} | {row.edge_count_median:.2f} | {row.edge_count_mean:.2f} | "
            f"{row.static_resolution_rate:.4f} |"
        )
    return (
        "# Phase 8R2A: Deterministic Static EVM CFG Dataset\n\n"
        "This phase constructs and validates graphs only. No classifier or GNN was trained, "
        "and no P5 model evaluation was performed.\n\n"
        "## Corpus\n\n"
        f"- Contracts/graphs: {audit['graphs_created']:,}\n"
        f"- Total nodes: {manifest['total_nodes']:,}\n"
        f"- Total edges: {manifest['total_edges']:,}\n"
        f"- Feature dimension: {manifest['node_feature_dimension']}\n"
        f"- Artifact chunks: {manifest['artifact_count']}\n"
        f"- Compressed artifact size: {manifest['artifact_size_bytes']:,} bytes\n"
        f"- Corpus hash: `{manifest['corpus_sha256']}`\n"
        f"- Zero-node graphs: {audit['zero_node_graphs']}\n"
        f"- Zero-edge graphs: {audit['zero_edge_graphs']}\n"
        f"- Non-finite features: {audit['non_finite_feature_count']}\n\n"
        "## Graph-Size Audit\n\n" + "\n".join(distribution_rows) + "\n\n"
        "## Jump Resolution\n\n"
        f"- Total JUMP/JUMPI: {audit['jump_total']:,}\n"
        f"- Statically resolved: {audit['jump_resolved']:,}\n"
        f"- Unresolved: {audit['jump_unresolved']:,}\n"
        f"- Invalid direct targets: {audit['jump_invalid_target']:,}\n"
        f"- Static resolution rate: {audit['static_resolution_rate']:.4f}\n\n"
        "Unresolved jumps receive no speculative destination edges. JUMPI retains only its "
        "sequential fall-through edge when its jump target is unresolved.\n\n"
        "## Period-Level Structural Audit\n\n" + "\n".join(period_rows) + "\n\n"
        "These are descriptive graph statistics, not model-performance measurements. P5 labels "
        "were not loaded or evaluated.\n\n"
        "## Frozen Representation\n\n"
        "The `phase8r2a-cfg-v1` block policy, 55-feature node order, three edge types, "
        "direct-PUSH jump rule, and deterministic chunk schema are frozen for Phase 8R2B. "
        "Exact PUSH operands are retained only in provenance arrays and are excluded from `x`.\n"
    )


def save_outputs(
    results: dict[str, Any],
    manifest: dict[str, Any],
    schema: dict[str, Any],
    graph_frame: pd.DataFrame,
    period_frame: pd.DataFrame,
    jump_frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "phase8r2a_results.json",
        "manifest": output_dir / "graph_manifest.json",
        "schema": output_dir / "graph_schema.json",
        "graphs": output_dir / "graph_statistics.csv",
        "periods": output_dir / "period_graph_statistics.csv",
        "jumps": output_dir / "jump_resolution_statistics.csv",
        "report": output_dir / "phase8r2a_report.md",
    }
    paths["results"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["schema"].write_text(json.dumps(schema, indent=2), encoding="utf-8")
    graph_frame.to_csv(paths["graphs"], index=False)
    period_frame.to_csv(paths["periods"], index=False)
    jump_frame.to_csv(paths["jumps"], index=False)
    paths["report"].write_text(build_report(results, manifest, period_frame), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = build_graph_corpus(config)
    for name, path in save_outputs(*outputs, config["output_dir"]).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
