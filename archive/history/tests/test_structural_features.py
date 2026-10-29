"""Contract-local structural feature tests."""

from __future__ import annotations

import math

import numpy as np

from src.features.opcode_categories import CATEGORY_NAMES, OPCODE_CATEGORIES
from src.features.structural import FEATURE_GROUPS, FEATURE_INDEX, FEATURE_NAMES, extract_structural_features, transform_structural_features


def _value(vector: np.ndarray, name: str) -> float:
    return float(vector[FEATURE_INDEX[name]])


def test_schema_is_stable_explicit_and_has_74_features() -> None:
    assert CATEGORY_NAMES == ("stack", "arithmetic", "comparison", "bitwise", "environment_context", "memory", "storage", "control_flow", "logging", "system_contract")
    assert set(OPCODE_CATEGORIES) == set(CATEGORY_NAMES)
    assert tuple(FEATURE_GROUPS) == ("A", "B", "C", "D", "E")
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 74


def test_extraction_is_deterministic_finite_and_label_independent() -> None:
    document = "PUSH1 PUSH2 ADD SLOAD CALL LOG1"
    first = extract_structural_features(document); second = extract_structural_features(document)
    assert np.array_equal(first, second) and np.isfinite(first).all()
    rows = transform_structural_features([document, document])
    assert (rows[0] != rows[1]).nnz == 0


def test_contract_locality() -> None:
    target = "PUSH1 ADD STOP"
    before = transform_structural_features([target, "CALL"])[0].toarray()
    after = transform_structural_features([target, "SELFDESTRUCT SELFDESTRUCT"])[0].toarray()
    assert np.array_equal(before, after)


def test_empty_null_and_unknown_tokens_are_safe() -> None:
    for document in (None, "", "   "):
        vector = extract_structural_features(document)
        assert np.isfinite(vector).all() and _value(vector, "opcode_count") == 0
        assert _value(vector, "has_opcode") == 0 and _value(vector, "unique_opcode_ratio") == 0
    unknown = extract_structural_features("FUTUREOP ADD")
    assert _value(unknown, "unknown_opcode_count") == 1
    assert _value(unknown, "unknown_opcode_ratio") == 0.5


def test_push_widths_and_stack_counts() -> None:
    vector = extract_structural_features("PUSH0 PUSH1 PUSH32 DUP1 DUP16 SWAP2 POP")
    assert _value(vector, "push_count") == 3
    assert _value(vector, "push_width_mean") == 11
    assert _value(vector, "push_width_max") == 32
    assert math.isclose(_value(vector, "push_width_std"), np.std([0, 1, 32]), rel_tol=1e-6)
    assert _value(vector, "dup_count") == 2 and _value(vector, "swap_count") == 1 and _value(vector, "pop_count") == 1


def test_category_selected_opcode_and_no_operand_content() -> None:
    first = extract_structural_features("PUSH1 0xaa CALL SSTORE LOG2")
    second = extract_structural_features("PUSH1 0xbb CALL SSTORE LOG2")
    assert np.array_equal(first, second)
    assert _value(first, "category_stack_count") == 1
    assert _value(first, "category_system_contract_count") == 1
    assert _value(first, "category_storage_count") == 1
    assert _value(first, "opcode_call_indicator") == 1
    assert _value(first, "opcode_log_family_count") == 1


def test_entropy_and_runs_match_hand_calculation() -> None:
    vector = extract_structural_features("ADD ADD MUL MUL MUL")
    expected_entropy = -(2 / 5 * math.log2(2 / 5) + 3 / 5 * math.log2(3 / 5))
    assert math.isclose(_value(vector, "opcode_entropy"), expected_entropy, rel_tol=1e-6)
    assert math.isclose(_value(vector, "normalized_opcode_entropy"), expected_entropy, rel_tol=1e-6)
    assert _value(vector, "maximum_identical_opcode_run_length") == 3
    assert _value(vector, "average_opcode_run_length") == 2.5
