"""Frozen label-independent features for creation and deployment-init bytecode."""

from __future__ import annotations

from collections import Counter
import re

import numpy as np

from src.extension.cfg import ContractCFG, Disassembly, build_cfg, disassemble
from src.extension.creation_code import RuntimeBoundary
from src.features.opcode_categories import CATEGORY_NAMES, OPCODE_CATEGORIES


IMPORTANT_OPCODES: tuple[str, ...] = (
    "SLOAD", "SSTORE", "CALL", "CALLCODE", "DELEGATECALL", "STATICCALL",
    "CREATE", "CREATE2", "SELFDESTRUCT", "CALLVALUE", "CALLER", "ORIGIN",
    "BALANCE", "SELFBALANCE", "LOG_FAMILY", "CODECOPY", "EXTCODECOPY",
    "RETURNDATACOPY", "RETURN", "REVERT",
)
OPERAND_FEATURES: tuple[str, ...] = (
    "zero_constant_count", "zero_constant_ratio", "small_constant_count",
    "small_constant_ratio", "unique_constant_count", "unique_constant_ratio",
    "repeated_constant_count", "repeated_constant_ratio",
    "selector_sized_4byte_count", "address_sized_20byte_count",
    "word_sized_32byte_count",
)


def _region_names(prefix: str, *, include_runtime_ratio: bool) -> tuple[str, ...]:
    size = (
        f"{prefix}_byte_length", f"{prefix}_instruction_count",
        f"{prefix}_unique_opcode_count", f"{prefix}_unique_opcode_ratio",
        f"{prefix}_opcode_entropy",
    ) + ((f"{prefix}_runtime_length_ratio",) if include_runtime_ratio else ())
    categories = tuple(
        f"{prefix}_category_{category}_{measure}"
        for category in CATEGORY_NAMES for measure in ("count", "ratio")
    )
    important = tuple(
        f"{prefix}_opcode_{opcode.lower()}_{measure}"
        for opcode in IMPORTANT_OPCODES for measure in ("count", "indicator")
    )
    stack = (
        f"{prefix}_push_count", f"{prefix}_push_ratio",
        f"{prefix}_mean_push_width", f"{prefix}_max_push_width",
        f"{prefix}_dup_count", f"{prefix}_swap_count", f"{prefix}_pop_count",
        f"{prefix}_truncated_push_count",
    )
    operands = tuple(f"{prefix}_{name}" for name in OPERAND_FEATURES)
    return size + categories + important + stack + operands


CREATION_FULL_FEATURE_NAMES: tuple[str, ...] = _region_names(
    "creation", include_runtime_ratio=True
)
INIT_BASE_FEATURE_NAMES: tuple[str, ...] = _region_names(
    "init", include_runtime_ratio=False
)
INIT_SUMMARY_FEATURE_NAMES: tuple[str, ...] = (
    "init_runtime_length_ratio", "init_basic_block_count",
    "init_control_flow_count", "init_call_family_count",
    "init_create_family_count", "init_return_revert_count",
)
DEPLOYMENT_INIT_FEATURE_NAMES: tuple[str, ...] = (
    INIT_BASE_FEATURE_NAMES + INIT_SUMMARY_FEATURE_NAMES
)
BOUNDARY_FEATURE_NAMES: tuple[str, ...] = (
    "boundary_runtime_exact_substring_found", "boundary_runtime_occurrence_count",
    "boundary_init_region_available", "boundary_runtime_match_ambiguous",
    "boundary_runtime_match_not_found", "boundary_runtime_fraction_of_creation",
    "boundary_trailing_byte_length", "boundary_trailing_nonempty",
)
COMBINED_FEATURE_NAMES: tuple[str, ...] = (
    CREATION_FULL_FEATURE_NAMES + DEPLOYMENT_INIT_FEATURE_NAMES + BOUNDARY_FEATURE_NAMES
)


def _operand_values(cfg: ContractCFG | Disassembly, small_constant_max: int) -> dict[str, float]:
    operands = [
        instruction.operand for instruction in cfg.instructions
        if instruction.push_width > 0 and not instruction.truncated
    ]
    count = len(operands)
    denominator = float(count) if count else 1.0
    integers = [int.from_bytes(value, "big") for value in operands]
    frequencies = Counter(integers)
    repeated = sum(amount - 1 for amount in frequencies.values() if amount > 1)
    widths = [len(value) for value in operands]
    return {
        "zero_constant_count": float(integers.count(0)),
        "zero_constant_ratio": integers.count(0) / denominator if count else 0.0,
        "small_constant_count": float(sum(value <= small_constant_max for value in integers)),
        "small_constant_ratio": sum(value <= small_constant_max for value in integers) / denominator if count else 0.0,
        "unique_constant_count": float(len(frequencies)),
        "unique_constant_ratio": len(frequencies) / denominator if count else 0.0,
        "repeated_constant_count": float(repeated),
        "repeated_constant_ratio": repeated / denominator if count else 0.0,
        "selector_sized_4byte_count": float(widths.count(4)),
        "address_sized_20byte_count": float(widths.count(20)),
        "word_sized_32byte_count": float(widths.count(32)),
    }


def extract_region_features(
    bytecode: object,
    *,
    prefix: str,
    runtime_length: int,
    include_runtime_ratio: bool,
    small_constant_max: int = 255,
) -> tuple[np.ndarray, Disassembly]:
    """Extract a finite fixed-order region vector without exposing constants."""
    cfg = disassemble(bytecode)
    code = b"" if cfg.invalid_hex else bytes().join(
        bytes([item.opcode]) + item.operand for item in cfg.instructions
    )
    names = [item.name for item in cfg.instructions]
    counts = Counter(names)
    total = len(names)
    denominator = float(total) if total else 1.0
    probabilities = np.asarray(list(counts.values()), dtype=float) / denominator
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if total else 0.0
    values: dict[str, float] = {
        f"{prefix}_byte_length": float(len(code)),
        f"{prefix}_instruction_count": float(total),
        f"{prefix}_unique_opcode_count": float(len(counts)),
        f"{prefix}_unique_opcode_ratio": len(counts) / denominator if total else 0.0,
        f"{prefix}_opcode_entropy": entropy,
    }
    if include_runtime_ratio:
        values[f"{prefix}_runtime_length_ratio"] = len(code) / runtime_length if runtime_length else 0.0
    for category, opcodes in OPCODE_CATEGORIES.items():
        amount = sum(counts[opcode] for opcode in opcodes)
        values[f"{prefix}_category_{category}_count"] = float(amount)
        values[f"{prefix}_category_{category}_ratio"] = amount / denominator if total else 0.0
    for opcode in IMPORTANT_OPCODES:
        amount = (
            sum(counts[f"LOG{index}"] for index in range(5))
            if opcode == "LOG_FAMILY" else counts[opcode]
        )
        stem = f"{prefix}_opcode_{opcode.lower()}"
        values[f"{stem}_count"] = float(amount)
        values[f"{stem}_indicator"] = float(amount > 0)
    widths = [item.push_width for item in cfg.instructions if item.push_width > 0]
    values.update({
        f"{prefix}_push_count": float(len(widths)),
        f"{prefix}_push_ratio": len(widths) / denominator if total else 0.0,
        f"{prefix}_mean_push_width": float(np.mean(widths)) if widths else 0.0,
        f"{prefix}_max_push_width": float(max(widths)) if widths else 0.0,
        f"{prefix}_dup_count": float(sum(amount for name, amount in counts.items() if re.fullmatch(r"DUP(?:[1-9]|1[0-6])", name))),
        f"{prefix}_swap_count": float(sum(amount for name, amount in counts.items() if re.fullmatch(r"SWAP(?:[1-9]|1[0-6])", name))),
        f"{prefix}_pop_count": float(counts["POP"]),
        f"{prefix}_truncated_push_count": float(cfg.truncated_push_count),
    })
    for name, value in _operand_values(cfg, small_constant_max).items():
        values[f"{prefix}_{name}"] = value
    ordered = _region_names(prefix, include_runtime_ratio=include_runtime_ratio)
    vector = np.asarray([values[name] for name in ordered], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Creation-code features must be finite.")
    return vector, cfg


def extract_creation_full_features(
    bytecode: object, runtime_length: int, *, small_constant_max: int = 255
) -> tuple[np.ndarray, Disassembly]:
    return extract_region_features(
        bytecode, prefix="creation", runtime_length=runtime_length,
        include_runtime_ratio=True, small_constant_max=small_constant_max,
    )


def extract_init_features(
    init_region: bytes | None, runtime_length: int, *, small_constant_max: int = 255
) -> tuple[np.ndarray, ContractCFG | None]:
    if init_region is None:
        return np.zeros(len(DEPLOYMENT_INIT_FEATURE_NAMES), dtype=np.float32), None
    base, _ = extract_region_features(
        "0x" + init_region.hex(), prefix="init", runtime_length=runtime_length,
        include_runtime_ratio=False, small_constant_max=small_constant_max,
    )
    cfg = build_cfg("0x" + init_region.hex())
    counts = Counter(item.name for item in cfg.instructions)
    summary = np.asarray([
        len(init_region) / runtime_length if runtime_length else 0.0,
        len(cfg.blocks),
        counts["JUMP"] + counts["JUMPI"] + counts["JUMPDEST"],
        sum(counts[name] for name in ("CALL", "CALLCODE", "DELEGATECALL", "STATICCALL")),
        counts["CREATE"] + counts["CREATE2"],
        counts["RETURN"] + counts["REVERT"],
    ], dtype=np.float32)
    vector = np.concatenate([base, summary]).astype(np.float32)
    if vector.shape != (len(DEPLOYMENT_INIT_FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Init features must be fixed and finite.")
    return vector, cfg


def boundary_feature_vector(boundary: RuntimeBoundary) -> np.ndarray:
    trailing = boundary.trailing_region
    values = np.asarray([
        boundary.runtime_exact_substring_found,
        boundary.runtime_occurrence_count,
        boundary.init_region_available,
        boundary.boundary_status == "ambiguous",
        boundary.boundary_status == "not_found",
        boundary.runtime_fraction_of_creation,
        len(trailing) if trailing is not None else 0,
        bool(trailing),
    ], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Boundary features must be finite.")
    return values
