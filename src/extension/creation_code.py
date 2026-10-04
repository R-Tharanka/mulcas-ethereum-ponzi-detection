"""Deterministic creation/runtime bytecode boundary analysis."""

from __future__ import annotations

from dataclasses import dataclass

from src.features.operands import decode_bytecode


@dataclass(frozen=True)
class RuntimeBoundary:
    creation: bytes | None
    runtime: bytes | None
    creation_parseable: bool
    runtime_parseable: bool
    runtime_exact_substring_found: bool
    runtime_occurrence_count: int
    first_runtime_substring_byte_offset: int | None
    last_runtime_substring_byte_offset: int | None
    bytes_before_first_runtime_match: int | None
    bytes_after_last_runtime_match: int | None
    runtime_fraction_of_creation: float
    init_region_available: bool
    init_region: bytes | None
    trailing_region: bytes | None
    boundary_status: str


def exact_occurrence_offsets(haystack: bytes, needle: bytes) -> tuple[int, ...]:
    """Return all exact offsets, including overlapping occurrences."""
    if not haystack or not needle:
        return ()
    offsets: list[int] = []
    start = 0
    while start <= len(haystack) - len(needle):
        offset = haystack.find(needle, start)
        if offset < 0:
            break
        offsets.append(offset)
        start = offset + 1
    return tuple(offsets)


def analyze_runtime_boundary(creation_code: object, runtime_code: object) -> RuntimeBoundary:
    """Locate runtime bytes conservatively without inventing an init boundary."""
    creation = decode_bytecode(creation_code)
    runtime = decode_bytecode(runtime_code)
    creation_ok = creation is not None
    runtime_ok = runtime is not None
    if not creation_ok:
        status = "invalid_creation"
        offsets: tuple[int, ...] = ()
    elif not runtime_ok:
        status = "invalid_runtime"
        offsets = ()
    elif not runtime:
        status = "empty_runtime"
        offsets = ()
    else:
        offsets = exact_occurrence_offsets(creation, runtime)
        status = "unique" if len(offsets) == 1 else "ambiguous" if offsets else "not_found"

    found = bool(offsets)
    first = offsets[0] if found else None
    last = offsets[-1] if found else None
    unique = len(offsets) == 1
    init_region = creation[:first] if unique and creation is not None and first is not None else None
    trailing = (
        creation[last + len(runtime) :]
        if found and creation is not None and runtime is not None and last is not None
        else None
    )
    creation_length = len(creation) if creation is not None else 0
    runtime_length = len(runtime) if runtime is not None else 0
    return RuntimeBoundary(
        creation=creation,
        runtime=runtime,
        creation_parseable=creation_ok,
        runtime_parseable=runtime_ok,
        runtime_exact_substring_found=found,
        runtime_occurrence_count=len(offsets),
        first_runtime_substring_byte_offset=first,
        last_runtime_substring_byte_offset=last,
        bytes_before_first_runtime_match=first,
        bytes_after_last_runtime_match=len(trailing) if trailing is not None else None,
        runtime_fraction_of_creation=(runtime_length / creation_length if creation_length else 0.0),
        init_region_available=unique,
        init_region=init_region,
        trailing_region=trailing,
        boundary_status=status,
    )
