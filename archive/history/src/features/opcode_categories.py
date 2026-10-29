"""Canonical opcode categories for deterministic structural features."""

from __future__ import annotations


def _numbered(prefix: str, start: int, end: int) -> frozenset[str]:
    return frozenset(f"{prefix}{index}" for index in range(start, end + 1))


OPCODE_CATEGORIES: dict[str, frozenset[str]] = {
    "stack": frozenset({"POP", "PUSH0"}) | _numbered("PUSH", 1, 32) | _numbered("DUP", 1, 16) | _numbered("SWAP", 1, 16),
    "arithmetic": frozenset({"ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD", "ADDMOD", "MULMOD", "EXP", "SIGNEXTEND"}),
    "comparison": frozenset({"LT", "GT", "SLT", "SGT", "EQ", "ISZERO"}),
    "bitwise": frozenset({"AND", "OR", "XOR", "NOT", "BYTE", "SHL", "SHR", "SAR"}),
    "environment_context": frozenset({"ADDRESS", "BALANCE", "ORIGIN", "CALLER", "CALLVALUE", "CALLDATALOAD", "CALLDATASIZE", "CALLDATACOPY", "CODESIZE", "CODECOPY", "GASPRICE", "EXTCODESIZE", "EXTCODECOPY", "RETURNDATASIZE", "RETURNDATACOPY", "EXTCODEHASH", "BLOCKHASH", "COINBASE", "TIMESTAMP", "NUMBER", "DIFFICULTY", "PREVRANDAO", "GASLIMIT", "CHAINID", "SELFBALANCE", "BASEFEE", "BLOBHASH", "BLOBBASEFEE"}),
    "memory": frozenset({"MLOAD", "MSTORE", "MSTORE8", "MSIZE", "MCOPY"}),
    "storage": frozenset({"SLOAD", "SSTORE", "TLOAD", "TSTORE"}),
    "control_flow": frozenset({"STOP", "JUMP", "JUMPI", "PC", "JUMPDEST", "GAS", "RETURN", "REVERT", "INVALID"}),
    "logging": _numbered("LOG", 0, 4),
    "system_contract": frozenset({"CREATE", "CALL", "CALLCODE", "DELEGATECALL", "CREATE2", "STATICCALL", "SELFDESTRUCT"}),
}

CATEGORY_NAMES: tuple[str, ...] = tuple(OPCODE_CATEGORIES)
KNOWN_OPCODES: frozenset[str] = frozenset().union(*OPCODE_CATEGORIES.values()) | frozenset({"SHA3", "KECCAK256"})

