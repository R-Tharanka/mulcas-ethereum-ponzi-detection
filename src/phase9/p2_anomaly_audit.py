"""Deterministic descriptive audit of the 251 frozen P2 false positives."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normal_opcode(value: Any) -> str:
    return " ".join(str(value).split())


def _normal_hex(value: Any) -> str:
    text = "".join(str(value).split()).lower()
    return text[2:] if text.startswith("0x") else text


def audit_p2_false_positives(
    raw: pd.DataFrame, inventory: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach exact hashes and summarize deterministic repeated FP groups."""
    fp = inventory.loc[
        (inventory["period"] == "P2") & (inventory["error_type"] == "false_positive")
    ].copy()
    if len(fp) != 251:
        raise ValueError(f"Expected 251 P2 false positives, found {len(fp)}.")
    raw_fields = raw[["row_index", "opcode", "contractCode"]]
    fp = fp.merge(raw_fields, on="row_index", how="left", validate="one_to_one")
    fp["normalized_opcode_sha256"] = fp["opcode"].map(lambda value: _sha256(_normal_opcode(value)))
    fp["normalized_runtime_sha256"] = fp["contractCode"].map(lambda value: _sha256(_normal_hex(value)))
    fp["runtime_byte_length"] = fp["contractCode"].map(lambda value: len(_normal_hex(value)) // 2)
    fp = fp.drop(columns=["opcode", "contractCode"])

    groups: list[dict[str, Any]] = []
    for group_type, column in (
        ("exact_opcode_hash", "normalized_opcode_sha256"),
        ("exact_runtime_hash", "normalized_runtime_sha256"),
        ("creator", "creator"),
    ):
        for value, part in fp.groupby(column, sort=True, dropna=False):
            if len(part) < 2:
                continue
            groups.append(
                {
                    "group_type": group_type,
                    "group_value": str(value),
                    "row_count": int(len(part)),
                    "first_block": int(part["createdBlockNumber"].min()),
                    "last_block": int(part["createdBlockNumber"].max()),
                    "unique_creators": int(part["creator"].nunique()),
                    "dominant_route": str(part["accepted_viewer"].mode().iloc[0]),
                    "mean_score": float(part["final_score"].mean()),
                }
            )
    group_frame = pd.DataFrame(groups).sort_values(
        ["group_type", "row_count", "group_value"], ascending=[True, False, True], kind="mergesort"
    ) if groups else pd.DataFrame(columns=["group_type", "group_value", "row_count"])
    summary = {
        "false_positive_count": int(len(fp)),
        "routing_counts": {str(k): int(v) for k, v in fp["accepted_viewer"].value_counts().items()},
        "previously_seen_creator_count": int(fp["previously_seen_creator"].sum()),
        "historical_opcode_clone_count": int(fp["has_historical_opcode_clone"].sum()),
        "historical_runtime_clone_count": int(fp["has_historical_runtime_bytecode_clone"].sum()),
        "repeated_group_count": int(len(group_frame)),
    }
    return fp, group_frame, summary
