"""Deterministic, contract-local EVM control-flow graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.features.operands import OPCODE_NAMES, decode_bytecode


EDGE_TYPES: dict[str, int] = {
    "FALL_THROUGH": 0,
    "STATIC_JUMP": 1,
    "CONDITIONAL_JUMP": 2,
}
TERMINATORS = frozenset({
    "STOP", "RETURN", "REVERT", "SELFDESTRUCT", "JUMP", "JUMPI", "INVALID"
})
BRANCHES = frozenset({"JUMP", "JUMPI"})


@dataclass(frozen=True)
class Instruction:
    pc: int
    opcode: int
    name: str
    push_width: int
    operand: bytes
    size: int
    truncated: bool
    supported: bool


@dataclass(frozen=True)
class Disassembly:
    instructions: tuple[Instruction, ...]
    invalid_hex: bool
    empty: bool
    truncated_push_count: int
    unsupported_opcode_count: int


@dataclass(frozen=True)
class BasicBlock:
    block_id: int
    start_pc: int
    end_pc: int
    instruction_count: int
    byte_length: int
    terminator_opcode: str
    instructions: tuple[Instruction, ...]


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    edge_type: int


@dataclass(frozen=True)
class JumpStatistics:
    total: int
    resolved: int
    unresolved: int
    invalid_target: int

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0


@dataclass(frozen=True)
class ContractCFG:
    instructions: tuple[Instruction, ...]
    blocks: tuple[BasicBlock, ...]
    edges: tuple[Edge, ...]
    jump_statistics: JumpStatistics
    invalid_hex: bool
    empty: bool
    truncated_push_count: int
    unsupported_opcode_count: int


def disassemble(bytecode: object) -> Disassembly:
    """Disassemble runtime bytecode without inferring missing operand bytes."""
    code = decode_bytecode(bytecode)
    if code is None:
        return Disassembly((), True, False, 0, 0)
    if not code:
        return Disassembly((), False, True, 0, 0)

    instructions: list[Instruction] = []
    pc = 0
    truncated_pushes = 0
    unsupported = 0
    while pc < len(code):
        start = pc
        opcode = code[pc]
        pc += 1
        supported = opcode in OPCODE_NAMES
        if not supported:
            unsupported += 1
        name = OPCODE_NAMES.get(opcode, f"UNKNOWN_0X{opcode:02X}")
        push_width = opcode - 0x5F if 0x60 <= opcode <= 0x7F else 0
        available = min(push_width, len(code) - pc)
        operand = code[pc : pc + available]
        truncated = available != push_width
        if truncated:
            truncated_pushes += 1
        pc += available
        instructions.append(
            Instruction(
                pc=start,
                opcode=opcode,
                name=name,
                push_width=push_width,
                operand=operand,
                size=1 + available,
                truncated=truncated,
                supported=supported,
            )
        )
    return Disassembly(tuple(instructions), False, False, truncated_pushes, unsupported)


def build_basic_blocks(instructions: tuple[Instruction, ...]) -> tuple[BasicBlock, ...]:
    """Partition every instruction exactly once using the frozen leader policy."""
    if not instructions:
        return ()
    leaders = {instructions[0].pc}
    for index, instruction in enumerate(instructions):
        if instruction.name == "JUMPDEST":
            leaders.add(instruction.pc)
        if instruction.name in TERMINATORS and index + 1 < len(instructions):
            leaders.add(instructions[index + 1].pc)

    grouped: list[tuple[Instruction, ...]] = []
    current: list[Instruction] = []
    for instruction in instructions:
        if current and instruction.pc in leaders:
            grouped.append(tuple(current))
            current = []
        current.append(instruction)
    if current:
        grouped.append(tuple(current))

    blocks: list[BasicBlock] = []
    for block_id, values in enumerate(grouped):
        final = values[-1]
        terminator = final.name if final.name in TERMINATORS else "FALL_THROUGH"
        blocks.append(
            BasicBlock(
                block_id=block_id,
                start_pc=values[0].pc,
                end_pc=final.pc + final.size - 1,
                instruction_count=len(values),
                byte_length=sum(item.size for item in values),
                terminator_opcode=terminator,
                instructions=values,
            )
        )
    if sum(block.instruction_count for block in blocks) != len(instructions):
        raise AssertionError("Every instruction must belong to exactly one basic block.")
    return tuple(blocks)


def _direct_target(block: BasicBlock, jumpdest_to_block: dict[int, int]) -> tuple[int | None, str]:
    """Resolve only a valid PUSH immediately preceding the branch."""
    if len(block.instructions) < 2:
        return None, "unresolved"
    candidate = block.instructions[-2]
    if candidate.push_width == 0 or candidate.truncated:
        return None, "unresolved"
    target_pc = int.from_bytes(candidate.operand, "big")
    if target_pc not in jumpdest_to_block:
        return None, "invalid_target"
    return jumpdest_to_block[target_pc], "resolved"


def build_edges(blocks: tuple[BasicBlock, ...]) -> tuple[tuple[Edge, ...], JumpStatistics]:
    jumpdest_to_block = {
        block.start_pc: block.block_id
        for block in blocks
        if block.instructions and block.instructions[0].name == "JUMPDEST"
    }
    edges: set[tuple[int, int, int]] = set()
    total = resolved = unresolved = invalid_target = 0
    for index, block in enumerate(blocks):
        terminator = block.terminator_opcode
        has_successor = index + 1 < len(blocks)
        if terminator in BRANCHES:
            total += 1
            target, status = _direct_target(block, jumpdest_to_block)
            if status == "resolved":
                resolved += 1
                edge_name = "STATIC_JUMP" if terminator == "JUMP" else "CONDITIONAL_JUMP"
                edges.add((block.block_id, int(target), EDGE_TYPES[edge_name]))
            elif status == "invalid_target":
                invalid_target += 1
            else:
                unresolved += 1
            if terminator == "JUMPI" and has_successor:
                edges.add((block.block_id, blocks[index + 1].block_id, EDGE_TYPES["FALL_THROUGH"]))
        elif terminator == "FALL_THROUGH" and has_successor:
            edges.add((block.block_id, blocks[index + 1].block_id, EDGE_TYPES["FALL_THROUGH"]))
    ordered = tuple(Edge(*values) for values in sorted(edges))
    return ordered, JumpStatistics(total, resolved, unresolved, invalid_target)


def build_cfg(bytecode: object) -> ContractCFG:
    """Construct one deterministic CFG using only the supplied contract bytecode."""
    parsed = disassemble(bytecode)
    blocks = build_basic_blocks(parsed.instructions)
    edges, jumps = build_edges(blocks)
    return ContractCFG(
        instructions=parsed.instructions,
        blocks=blocks,
        edges=edges,
        jump_statistics=jumps,
        invalid_hex=parsed.invalid_hex,
        empty=parsed.empty,
        truncated_push_count=parsed.truncated_push_count,
        unsupported_opcode_count=parsed.unsupported_opcode_count,
    )

