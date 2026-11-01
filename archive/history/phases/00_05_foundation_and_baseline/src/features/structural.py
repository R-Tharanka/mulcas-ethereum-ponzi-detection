"""Fit-free, opcode-name-only structural contract features."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from src.features.opcode_categories import CATEGORY_NAMES, KNOWN_OPCODES, OPCODE_CATEGORIES


SELECTED_OPCODES: tuple[str, ...] = (
    "CALL", "CALLCODE", "DELEGATECALL", "STATICCALL", "SLOAD", "SSTORE",
    "JUMP", "JUMPI", "JUMPDEST", "CREATE", "CREATE2", "SELFDESTRUCT",
    "CALLVALUE", "CALLER", "ORIGIN", "BALANCE", "SELFBALANCE",
)
GROUPS: tuple[str, ...] = ("A", "B", "C", "D", "E")
PUSH_PATTERN = re.compile(r"^PUSH(?:0|[1-9]|[12][0-9]|3[0-2])$")
OPERAND_LITERAL_PATTERN = re.compile(r"^0x[0-9a-f]+$", re.IGNORECASE)


def _schema() -> dict[str, tuple[str, ...]]:
    group_a = ("has_opcode", "opcode_count", "unique_opcode_count", "unique_opcode_ratio", "unknown_opcode_count", "unknown_opcode_ratio")
    group_b = tuple(name for category in CATEGORY_NAMES for name in (f"category_{category}_count", f"category_{category}_ratio"))
    group_c = tuple(name for opcode in SELECTED_OPCODES for name in (f"opcode_{opcode.lower()}_count", f"opcode_{opcode.lower()}_indicator")) + ("opcode_log_family_count", "opcode_log_family_indicator")
    group_d = ("push_count", "push_ratio", "push_width_mean", "push_width_max", "push_width_std", "dup_count", "swap_count", "pop_count")
    group_e = ("opcode_entropy", "normalized_opcode_entropy", "maximum_identical_opcode_run_length", "average_opcode_run_length")
    return {"A": group_a, "B": group_b, "C": group_c, "D": group_d, "E": group_e}


FEATURE_GROUPS: dict[str, tuple[str, ...]] = _schema()
FEATURE_NAMES: tuple[str, ...] = tuple(name for group in GROUPS for name in FEATURE_GROUPS[group])
FEATURE_INDEX: dict[str, int] = {name: index for index, name in enumerate(FEATURE_NAMES)}


def tokenize_opcode_names(document: object) -> list[str]:
    """Return uppercase opcode-name tokens without interpreting operand content."""
    if document is None or (isinstance(document, float) and math.isnan(document)):
        return []
    text = str(document).strip()
    if not text:
        return []
    return [
        token.upper()
        for token in text.split()
        if not OPERAND_LITERAL_PATTERN.fullmatch(token)
    ]


def _run_lengths(tokens: Sequence[str]) -> list[int]:
    if not tokens:
        return []
    lengths, current = [], 1
    for previous, token in zip(tokens, tokens[1:]):
        if token == previous:
            current += 1
        else:
            lengths.append(current); current = 1
    lengths.append(current)
    return lengths


def extract_structural_features(document: object) -> np.ndarray:
    tokens = tokenize_opcode_names(document)
    counts = Counter(tokens)
    total = len(tokens)
    denominator = float(total) if total else 1.0
    values: dict[str, float] = {
        "has_opcode": float(bool(total)), "opcode_count": float(total),
        "unique_opcode_count": float(len(counts)),
        "unique_opcode_ratio": len(counts) / denominator if total else 0.0,
    }
    unknown = sum(count for opcode, count in counts.items() if opcode not in KNOWN_OPCODES)
    values.update({"unknown_opcode_count": float(unknown), "unknown_opcode_ratio": unknown / denominator if total else 0.0})
    for category, opcodes in OPCODE_CATEGORIES.items():
        count = sum(counts[opcode] for opcode in opcodes)
        values[f"category_{category}_count"] = float(count)
        values[f"category_{category}_ratio"] = count / denominator if total else 0.0
    for opcode in SELECTED_OPCODES:
        count = counts[opcode]
        values[f"opcode_{opcode.lower()}_count"] = float(count)
        values[f"opcode_{opcode.lower()}_indicator"] = float(count > 0)
    log_count = sum(counts[f"LOG{index}"] for index in range(5))
    values.update({"opcode_log_family_count": float(log_count), "opcode_log_family_indicator": float(log_count > 0)})
    widths = [int(token[4:]) for token in tokens if PUSH_PATTERN.fullmatch(token)]
    values.update({
        "push_count": float(len(widths)), "push_ratio": len(widths) / denominator if total else 0.0,
        "push_width_mean": float(np.mean(widths)) if widths else 0.0,
        "push_width_max": float(max(widths)) if widths else 0.0,
        "push_width_std": float(np.std(widths)) if widths else 0.0,
        "dup_count": float(sum(count for opcode, count in counts.items() if re.fullmatch(r"DUP(?:[1-9]|1[0-6])", opcode))),
        "swap_count": float(sum(count for opcode, count in counts.items() if re.fullmatch(r"SWAP(?:[1-9]|1[0-6])", opcode))),
        "pop_count": float(counts["POP"]),
    })
    if total:
        probabilities = np.asarray(list(counts.values()), dtype=float) / total
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        normalized = entropy / math.log2(len(counts)) if len(counts) > 1 else 0.0
    else:
        entropy = normalized = 0.0
    runs = _run_lengths(tokens)
    values.update({
        "opcode_entropy": entropy, "normalized_opcode_entropy": normalized,
        "maximum_identical_opcode_run_length": float(max(runs)) if runs else 0.0,
        "average_opcode_run_length": float(np.mean(runs)) if runs else 0.0,
    })
    vector = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Structural features must be finite.")
    return vector


def transform_structural_features(documents: Iterable[object], groups: Sequence[str] = GROUPS) -> csr_matrix:
    selected = [FEATURE_INDEX[name] for group in groups for name in FEATURE_GROUPS[group]]
    rows = [extract_structural_features(document)[selected] for document in documents]
    return csr_matrix(np.vstack(rows) if rows else np.empty((0, len(selected)), dtype=np.float32))


def feature_names_for_groups(groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for group in groups for name in FEATURE_GROUPS[group])
