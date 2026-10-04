from __future__ import annotations

import numpy as np

from src.extension.cfg import build_cfg
from src.extension.graph_features import (
    FEATURE_DIMENSION,
    FEATURE_INDEX,
    FEATURE_NAMES,
    extract_block_features,
    graph_arrays,
)


def test_feature_schema_is_stable_finite_and_identity_free() -> None:
    assert FEATURE_DIMENSION == 55
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    forbidden = ("label", "address_identity", "creator", "selector_identity", "constant_identity")
    assert not any(term in name for name in FEATURE_NAMES for term in forbidden)
    x, edge_index, edge_type = graph_arrays(build_cfg("0x60016002015500"))
    assert x.shape[1] == FEATURE_DIMENSION
    assert np.isfinite(x).all()
    assert edge_index.shape[0] == 2
    assert len(edge_type) == edge_index.shape[1]


def test_features_reuse_categories_and_hand_check_counts() -> None:
    block = build_cfg("0x6001545560010100").blocks[0]
    values = extract_block_features(block)
    assert values[FEATURE_INDEX["instruction_count"]] == 6
    assert values[FEATURE_INDEX["push_count"]] == 2
    assert values[FEATURE_INDEX["opcode_sload_count"]] == 1
    assert values[FEATURE_INDEX["opcode_sstore_count"]] == 1
    assert values[FEATURE_INDEX["category_storage_count"]] == 2
    assert values[FEATURE_INDEX["terminator_stop"]] == 1


def test_exact_push_values_do_not_change_node_features() -> None:
    first = graph_arrays(build_cfg("0x63aabbccdd00"))[0]
    second = graph_arrays(build_cfg("0x631122334400"))[0]
    np.testing.assert_array_equal(first, second)


def test_graph_is_contract_local_and_label_independent() -> None:
    code = "0x6003565b00"
    original = graph_arrays(build_cfg(code))
    unrelated = graph_arrays(build_cfg("0x600160020100"))
    repeated = graph_arrays(build_cfg(code))
    assert unrelated[0].shape != (0, FEATURE_DIMENSION)
    for left, right in zip(original, repeated, strict=True):
        np.testing.assert_array_equal(left, right)

