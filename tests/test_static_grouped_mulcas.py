from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.experiments.static_grouped_mulcas import (
    EXPECTED_HISTORICAL_SHA256,
    HISTORICAL_RESULT,
    add_duplicate_groups,
    grouped_cv_splits,
    grouped_split,
    sha256_file,
)


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for index in range(100):
        rows.append(
            {
                "row_index": index,
                "address": f"0x{index:040x}",
                "opcode": f"PUSH1 token_{index}",
                "label": int(index % 10 == 0),
            }
        )
    # A--B share address and B--C share opcode, so all three must stay together.
    rows[1]["address"] = rows[0]["address"]
    rows[2]["opcode"] = rows[1]["opcode"]
    return add_duplicate_groups(pd.DataFrame(rows))


def test_duplicate_groups_are_transitive():
    frame = _synthetic_frame()
    assert frame.loc[0, "duplicate_group"] == frame.loc[1, "duplicate_group"]
    assert frame.loc[1, "duplicate_group"] == frame.loc[2, "duplicate_group"]


def test_outer_and_inner_splits_isolate_groups_addresses_and_opcodes():
    parts = grouped_split(_synthetic_frame())
    for left, right in ((parts["fit"], parts["validation"]), (parts["development"], parts["test"])):
        assert set(left.duplicate_group).isdisjoint(right.duplicate_group)
        assert set(left.normalized_address).isdisjoint(right.normalized_address)
        assert set(left.opcode_sha256).isdisjoint(right.opcode_sha256)


def test_grouped_cv_never_splits_a_duplicate_group():
    frame = _synthetic_frame()
    for train_idx, test_idx in grouped_cv_splits(frame):
        assert set(frame.iloc[train_idx].duplicate_group).isdisjoint(
            frame.iloc[test_idx].duplicate_group
        )


def test_historical_result_is_preserved_and_is_not_successor_output():
    assert sha256_file(HISTORICAL_RESULT) == EXPECTED_HISTORICAL_SHA256
    payload = json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))
    assert payload["experiment"] == "random_stratified_static_opcode_three_viewer"
    assert "duplicate_group_safe" not in payload["experiment"]


def test_successor_result_paths_are_separate():
    source = Path("src/experiments/static_grouped_mulcas.py").read_text(encoding="utf-8")
    assert 'RESULT_DIR = ROOT / "results" / "experiments" / "static_grouped_mulcas"' in source
    assert "HISTORICAL_RESULT.write" not in source
