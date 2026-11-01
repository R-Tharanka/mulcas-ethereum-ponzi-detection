"""Governance rules for external label provenance and Track B feasibility."""
from __future__ import annotations


def track_b_rating(new_positive: int, provenance: str, benign_usable: bool,
                   overlap_free: bool, temporal_usable: bool) -> str:
    if provenance not in {"A", "B"} or new_positive <= 0:
        return "NONE"
    if new_positive >= 100 and benign_usable and overlap_free and temporal_usable:
        return "STRONG"
    if 30 <= new_positive <= 99 and overlap_free:
        return "MODERATE"
    if 1 <= new_positive <= 29:
        return "WEAK"
    return "NONE"


def complete_benchmark(provenance: str, positive_only: bool, benign_usable: bool) -> bool:
    return provenance in {"A", "B"} and not positive_only and benign_usable


def ground_truth_eligible(provenance: str) -> bool:
    return provenance in {"A", "B"}
