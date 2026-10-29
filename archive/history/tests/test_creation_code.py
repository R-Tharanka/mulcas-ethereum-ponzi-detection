from __future__ import annotations

from src.extension.creation_code import analyze_runtime_boundary, exact_occurrence_offsets


def test_exact_runtime_substring_unique_boundary() -> None:
    result = analyze_runtime_boundary("0x600160026003", "0x6002")
    assert result.runtime_occurrence_count == 1
    assert result.first_runtime_substring_byte_offset == 2
    assert result.last_runtime_substring_byte_offset == 2
    assert result.init_region_available is True
    assert result.init_region == bytes.fromhex("6001")
    assert result.trailing_region == bytes.fromhex("6003")


def test_ambiguous_runtime_is_never_an_init_boundary() -> None:
    result = analyze_runtime_boundary("0xaabbaabb", "0xaabb")
    assert exact_occurrence_offsets(b"aaaa", b"aaa") == (0, 1)
    assert result.runtime_occurrence_count == 2
    assert result.boundary_status == "ambiguous"
    assert result.init_region_available is False
    assert result.init_region is None
    assert result.first_runtime_substring_byte_offset == 0
    assert result.last_runtime_substring_byte_offset == 2


def test_missing_and_invalid_runtime_never_invent_boundaries() -> None:
    missing = analyze_runtime_boundary("0x60016002", "0x6003")
    invalid = analyze_runtime_boundary("0x60016002", "not-hex")
    assert missing.boundary_status == "not_found"
    assert invalid.boundary_status == "invalid_runtime"
    assert missing.init_region is None and invalid.init_region is None
    assert missing.first_runtime_substring_byte_offset is None


def test_boundary_analysis_is_deterministic() -> None:
    first = analyze_runtime_boundary("0x600160026003", "0x6002")
    second = analyze_runtime_boundary("0x600160026003", "0x6002")
    assert first == second
