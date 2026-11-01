from __future__ import annotations

import numpy as np

from src.extension.creation_code import analyze_runtime_boundary
from src.extension.creation_features import (
    BOUNDARY_FEATURE_NAMES,
    COMBINED_FEATURE_NAMES,
    CREATION_FULL_FEATURE_NAMES,
    DEPLOYMENT_INIT_FEATURE_NAMES,
    boundary_feature_vector,
    extract_creation_full_features,
    extract_init_features,
)


def test_frozen_schema_dimensions_order_and_no_raw_constants() -> None:
    assert len(CREATION_FULL_FEATURE_NAMES) == len(set(CREATION_FULL_FEATURE_NAMES)) == 85
    assert len(DEPLOYMENT_INIT_FEATURE_NAMES) == len(set(DEPLOYMENT_INIT_FEATURE_NAMES)) == 90
    assert len(COMBINED_FEATURE_NAMES) == len(set(COMBINED_FEATURE_NAMES)) == 183
    assert COMBINED_FEATURE_NAMES == CREATION_FULL_FEATURE_NAMES + DEPLOYMENT_INIT_FEATURE_NAMES + BOUNDARY_FEATURE_NAMES
    forbidden = ("0x", "exact_constant", "address_identity", "selector_identity")
    assert not any(term in name for name in COMBINED_FEATURE_NAMES for term in forbidden)


def test_full_features_are_finite_and_hand_checkable() -> None:
    vector, cfg = extract_creation_full_features("0x600060013955f3", runtime_length=2)
    index = {name: position for position, name in enumerate(CREATION_FULL_FEATURE_NAMES)}
    assert np.isfinite(vector).all()
    assert vector[index["creation_byte_length"]] == 7
    assert vector[index["creation_instruction_count"]] == 5
    assert vector[index["creation_push_count"]] == 2
    assert vector[index["creation_opcode_codecopy_count"]] == 1
    assert vector[index["creation_opcode_sstore_count"]] == 1
    assert len(cfg.instructions) == 5


def test_unavailable_init_is_zero_not_fabricated() -> None:
    vector, cfg = extract_init_features(None, runtime_length=10)
    assert cfg is None
    assert vector.shape == (90,)
    assert np.count_nonzero(vector) == 0


def test_init_and_boundary_features_are_finite() -> None:
    boundary = analyze_runtime_boundary("0x60016002396000f3", "0x6000f3")
    vector, cfg = extract_init_features(boundary.init_region, runtime_length=3)
    assert cfg is not None
    assert np.isfinite(vector).all()
    assert np.isfinite(boundary_feature_vector(boundary)).all()


def test_contract_features_are_local_and_label_independent() -> None:
    first, _ = extract_creation_full_features("0x6001600201", runtime_length=2)
    _other, _ = extract_creation_full_features("0x7f" + "aa" * 32, runtime_length=33)
    repeated, _ = extract_creation_full_features("0x6001600201", runtime_length=2)
    np.testing.assert_array_equal(first, repeated)
