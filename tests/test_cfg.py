from __future__ import annotations

from src.extension.cfg import EDGE_TYPES, build_cfg, disassemble


def _edges(code: str) -> set[tuple[int, int, int]]:
    return {(edge.source, edge.target, edge.edge_type) for edge in build_cfg(code).edges}


def test_straight_line_and_stop_form_one_ordered_block() -> None:
    cfg = build_cfg("0x600160020100")
    assert [(item.pc, item.name, item.push_width, item.operand, item.size) for item in cfg.instructions] == [
        (0, "PUSH1", 1, b"\x01", 2),
        (2, "PUSH1", 1, b"\x02", 2),
        (4, "ADD", 0, b"", 1),
        (5, "STOP", 0, b"", 1),
    ]
    assert [(block.start_pc, block.end_pc, block.instruction_count, block.byte_length, block.terminator_opcode) for block in cfg.blocks] == [
        (0, 5, 4, 6, "STOP")
    ]
    assert not cfg.edges


def test_direct_jump_resolves_only_to_jumpdest() -> None:
    cfg = build_cfg("0x6003565b00")
    assert [(block.start_pc, block.end_pc) for block in cfg.blocks] == [(0, 2), (3, 4)]
    assert _edges("0x6003565b00") == {(0, 1, EDGE_TYPES["STATIC_JUMP"])}
    assert cfg.jump_statistics.resolved == 1
    assert cfg.jump_statistics.resolution_rate == 1.0


def test_jumpi_has_resolved_target_and_fallthrough() -> None:
    edges = _edges("0x6003575b00")
    assert edges == {
        (0, 1, EDGE_TYPES["CONDITIONAL_JUMP"]),
        (0, 1, EDGE_TYPES["FALL_THROUGH"]),
    }


def test_multiple_jumpdest_blocks_preserve_order() -> None:
    cfg = build_cfg("0x5b6006565b005b00")
    assert [block.start_pc for block in cfg.blocks] == [0, 4, 6]
    assert [instruction.pc for block in cfg.blocks for instruction in block.instructions] == [
        instruction.pc for instruction in cfg.instructions
    ]


def test_unresolved_and_invalid_jumps_never_receive_fabricated_edges() -> None:
    unresolved = build_cfg("0x565b00")
    invalid = build_cfg("0x60ff565b00")
    assert unresolved.jump_statistics.unresolved == 1
    assert invalid.jump_statistics.invalid_target == 1
    assert not any(edge.edge_type == EDGE_TYPES["STATIC_JUMP"] for edge in unresolved.edges)
    assert not any(edge.edge_type == EDGE_TYPES["STATIC_JUMP"] for edge in invalid.edges)


def test_return_revert_and_invalid_terminate_blocks_without_fallthrough() -> None:
    cfg = build_cfg("0xf36000fdfe")
    assert [block.terminator_opcode for block in cfg.blocks] == ["RETURN", "REVERT", "INVALID"]
    assert not cfg.edges


def test_truncated_push_unsupported_opcode_and_empty_are_explicit() -> None:
    truncated = disassemble("0x61aa")
    unsupported = disassemble("0x0c00")
    empty = build_cfg("0x")
    assert truncated.truncated_push_count == 1
    assert truncated.instructions[0].operand == b"\xaa"
    assert truncated.instructions[0].size == 2
    assert unsupported.instructions[0].name == "UNKNOWN_0X0C"
    assert unsupported.unsupported_opcode_count == 1
    assert empty.empty and not empty.blocks and not empty.edges


def test_cfg_construction_is_deterministic() -> None:
    code = "0x6003575b6006565b00"
    assert build_cfg(code) == build_cfg(code)

