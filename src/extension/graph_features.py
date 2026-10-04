"""Frozen block-local feature schema for Phase 8R2A control-flow graphs."""

from __future__ import annotations

from collections import Counter
import re

import numpy as np

from src.extension.cfg import BasicBlock, ContractCFG
from src.features.opcode_categories import CATEGORY_NAMES, OPCODE_CATEGORIES


IMPORTANT_OPCODES: tuple[str, ...] = (
    "SLOAD", "SSTORE", "CALL", "CALLCODE", "DELEGATECALL", "STATICCALL",
    "CREATE", "CREATE2", "SELFDESTRUCT", "CALLVALUE", "CALLER", "ORIGIN",
    "BALANCE", "SELFBALANCE",
)
TERMINATOR_FEATURES: tuple[str, ...] = (
    "fall_through", "jump", "jumpi", "stop", "return", "revert",
    "selfdestruct", "other_terminal",
)


def _feature_names() -> tuple[str, ...]:
    size = (
        "instruction_count", "byte_length", "unique_opcode_count",
        "unique_opcode_ratio", "opcode_entropy",
    )
    categories = tuple(
        name
        for category in CATEGORY_NAMES
        for name in (f"category_{category}_count", f"category_{category}_ratio")
    )
    important = tuple(f"opcode_{name.lower()}_count" for name in IMPORTANT_OPCODES) + (
        "opcode_log_family_count",
    )
    stack = (
        "push_count", "push_ratio", "mean_push_width", "max_push_width",
        "dup_count", "swap_count", "pop_count",
    )
    terminators = tuple(f"terminator_{name}" for name in TERMINATOR_FEATURES)
    return size + categories + important + stack + terminators


FEATURE_NAMES: tuple[str, ...] = _feature_names()
FEATURE_INDEX: dict[str, int] = {name: index for index, name in enumerate(FEATURE_NAMES)}
FEATURE_DIMENSION = len(FEATURE_NAMES)


def _terminator_group(name: str) -> str:
    lowered = name.lower()
    if lowered in TERMINATOR_FEATURES[:-1]:
        return lowered
    return "other_terminal"


def extract_block_features(block: BasicBlock) -> np.ndarray:
    """Return fixed-order finite features without exposing exact operand values."""
    names = [item.name for item in block.instructions]
    counts = Counter(names)
    total = len(names)
    denominator = float(total) if total else 1.0
    probabilities = np.asarray(list(counts.values()), dtype=float) / denominator
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if total else 0.0
    values: dict[str, float] = {
        "instruction_count": float(total),
        "byte_length": float(block.byte_length),
        "unique_opcode_count": float(len(counts)),
        "unique_opcode_ratio": len(counts) / denominator if total else 0.0,
        "opcode_entropy": entropy,
    }
    for category, opcodes in OPCODE_CATEGORIES.items():
        count = sum(counts[opcode] for opcode in opcodes)
        values[f"category_{category}_count"] = float(count)
        values[f"category_{category}_ratio"] = count / denominator if total else 0.0
    for opcode in IMPORTANT_OPCODES:
        values[f"opcode_{opcode.lower()}_count"] = float(counts[opcode])
    values["opcode_log_family_count"] = float(sum(counts[f"LOG{index}"] for index in range(5)))

    widths = [item.push_width for item in block.instructions if item.push_width > 0]
    values.update({
        "push_count": float(len(widths)),
        "push_ratio": len(widths) / denominator if total else 0.0,
        "mean_push_width": float(np.mean(widths)) if widths else 0.0,
        "max_push_width": float(max(widths)) if widths else 0.0,
        "dup_count": float(sum(count for name, count in counts.items() if re.fullmatch(r"DUP(?:[1-9]|1[0-6])", name))),
        "swap_count": float(sum(count for name, count in counts.items() if re.fullmatch(r"SWAP(?:[1-9]|1[0-6])", name))),
        "pop_count": float(counts["POP"]),
    })
    terminator = _terminator_group(block.terminator_opcode)
    for name in TERMINATOR_FEATURES:
        values[f"terminator_{name}"] = float(name == terminator)

    vector = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
        raise ValueError("Block feature vector must have fixed finite dimensions.")
    return vector


def graph_arrays(cfg: ContractCFG) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a CFG into framework-neutral node and edge arrays."""
    x = (
        np.vstack([extract_block_features(block) for block in cfg.blocks]).astype(np.float32)
        if cfg.blocks else np.empty((0, FEATURE_DIMENSION), dtype=np.float32)
    )
    edge_index = (
        np.asarray([[edge.source, edge.target] for edge in cfg.edges], dtype=np.int64).T
        if cfg.edges else np.empty((2, 0), dtype=np.int64)
    )
    edge_type = np.asarray([edge.edge_type for edge in cfg.edges], dtype=np.int8)
    return x, edge_index, edge_type
