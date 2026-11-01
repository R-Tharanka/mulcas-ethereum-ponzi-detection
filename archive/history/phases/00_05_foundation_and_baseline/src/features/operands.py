"""Deterministic normalized features from deployed EVM PUSH operands."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import csr_matrix


def _opcode_names() -> dict[int, str]:
    names = {
        0x00:"STOP",0x01:"ADD",0x02:"MUL",0x03:"SUB",0x04:"DIV",0x05:"SDIV",0x06:"MOD",0x07:"SMOD",0x08:"ADDMOD",0x09:"MULMOD",0x0A:"EXP",0x0B:"SIGNEXTEND",
        0x10:"LT",0x11:"GT",0x12:"SLT",0x13:"SGT",0x14:"EQ",0x15:"ISZERO",0x16:"AND",0x17:"OR",0x18:"XOR",0x19:"NOT",0x1A:"BYTE",0x1B:"SHL",0x1C:"SHR",0x1D:"SAR",0x20:"SHA3",
        0x30:"ADDRESS",0x31:"BALANCE",0x32:"ORIGIN",0x33:"CALLER",0x34:"CALLVALUE",0x35:"CALLDATALOAD",0x36:"CALLDATASIZE",0x37:"CALLDATACOPY",0x38:"CODESIZE",0x39:"CODECOPY",0x3A:"GASPRICE",0x3B:"EXTCODESIZE",0x3C:"EXTCODECOPY",0x3D:"RETURNDATASIZE",0x3E:"RETURNDATACOPY",0x3F:"EXTCODEHASH",
        0x40:"BLOCKHASH",0x41:"COINBASE",0x42:"TIMESTAMP",0x43:"NUMBER",0x44:"DIFFICULTY",0x45:"GASLIMIT",0x46:"CHAINID",0x47:"SELFBALANCE",
        0x50:"POP",0x51:"MLOAD",0x52:"MSTORE",0x53:"MSTORE8",0x54:"SLOAD",0x55:"SSTORE",0x56:"JUMP",0x57:"JUMPI",0x58:"GETPC",0x59:"MSIZE",0x5A:"GAS",0x5B:"JUMPDEST",
        0xF0:"CREATE",0xF1:"CALL",0xF2:"CALLCODE",0xF3:"RETURN",0xF4:"DELEGATECALL",0xF5:"CREATE2",0xFA:"STATICCALL",0xFD:"REVERT",0xFE:"INVALID",0xFF:"SELFDESTRUCT",
    }
    names.update({value: f"PUSH{value - 0x5F}" for value in range(0x60, 0x80)})
    names.update({value: f"DUP{value - 0x7F}" for value in range(0x80, 0x90)})
    names.update({value: f"SWAP{value - 0x8F}" for value in range(0x90, 0xA0)})
    names.update({value: f"LOG{value - 0xA0}" for value in range(0xA0, 0xA5)})
    return names


OPCODE_NAMES = _opcode_names()
GROUPS: tuple[str, ...] = ("A", "B", "C", "D")
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "A": ("operand_instruction_count", "push_operand_count", "parseable_operand_count", "parseable_operand_ratio", "total_operand_bytes", "mean_operand_bytes", "max_operand_bytes"),
    "B": ("zero_constant_count", "zero_constant_ratio", "one_constant_count", "one_constant_ratio", "small_constant_count", "small_constant_ratio", "unique_constant_count", "unique_constant_ratio", "repeated_constant_count", "repeated_constant_ratio"),
    "C": ("operand_width_mean", "operand_width_median", "operand_width_std", "operand_width_max", "width_1_count", "width_1_ratio", "width_2_4_count", "width_2_4_ratio", "width_5_20_count", "width_5_20_ratio", "width_21_32_count", "width_21_32_ratio"),
    "D": ("selector_sized_4byte_count", "selector_sized_4byte_ratio", "address_sized_20byte_count", "address_sized_20byte_ratio", "word_sized_32byte_count", "word_sized_32byte_ratio", "repeated_selector_sized_count", "repeated_address_sized_count"),
}
FEATURE_NAMES: tuple[str, ...] = tuple(name for group in GROUPS for name in FEATURE_GROUPS[group])
FEATURE_INDEX = {name:index for index,name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class OperandParseResult:
    instruction_count: int
    push_instruction_count: int
    operands: tuple[bytes, ...]
    malformed_truncated_push_count: int
    unsupported_opcode_count: int
    null_or_empty: bool


def decode_bytecode(bytecode: object) -> bytes | None:
    if bytecode is None or (isinstance(bytecode, float) and math.isnan(bytecode)):
        return None
    text = str(bytecode).strip()
    if text.startswith(("0x", "0X")): text = text[2:]
    if not text: return b""
    if len(text) % 2 or any(character not in "0123456789abcdefABCDEF" for character in text): return None
    return bytes.fromhex(text)


def disassemble_operands(bytecode: object) -> OperandParseResult:
    code = decode_bytecode(bytecode)
    if code is None or not code:
        return OperandParseResult(0, 0, (), 0, 0, True)
    pc=0; instructions=0; pushes=0; malformed=0; unsupported=0; operands=[]
    while pc < len(code):
        opcode=code[pc]; pc += 1; instructions += 1
        if opcode not in OPCODE_NAMES: unsupported += 1
        width = opcode - 0x5F if 0x60 <= opcode <= 0x7F else None
        if width is None: continue
        pushes += 1
        if pc + width > len(code): malformed += 1; pc = len(code); continue
        operands.append(code[pc:pc+width]); pc += width
    return OperandParseResult(instructions, pushes, tuple(operands), malformed, unsupported, False)


def disassemble_opcode_names(bytecode: object) -> tuple[str, ...]:
    """Disassemble names with the same PUSH-byte consumption policy."""
    code = decode_bytecode(bytecode)
    if code is None:
        return ()
    names=[]; pc=0
    while pc < len(code):
        opcode=code[pc]; pc += 1
        width=opcode-0x5F if 0x60<=opcode<=0x7F else 0
        if pc+width>len(code): break
        names.append(OPCODE_NAMES.get(opcode, "INVALID")); pc+=width
    return tuple(names)


def extract_operand_features(bytecode: object, *, small_constant_max: int = 255) -> np.ndarray:
    if small_constant_max < 1: raise ValueError("small_constant_max must be positive")
    parsed=disassemble_operands(bytecode); operands=parsed.operands; count=len(operands); denominator=float(count) if count else 1.0
    widths=np.asarray([len(value) for value in operands], dtype=float)
    integers=[int.from_bytes(value, "big") for value in operands]
    frequencies=Counter(integers); repeated=sum(value-1 for value in frequencies.values() if value>1)
    def bucket(low:int, high:int)->int: return sum(low <= width <= high for width in widths)
    def repeated_width(width:int)->int:
        selected=Counter(value for value in operands if len(value)==width)
        return sum(amount-1 for amount in selected.values() if amount>1)
    zero=integers.count(0); one=integers.count(1); small=sum(value <= small_constant_max for value in integers)
    values={
        "operand_instruction_count":float(parsed.push_instruction_count), "push_operand_count":float(parsed.push_instruction_count),
        "parseable_operand_count":float(count), "parseable_operand_ratio":count/parsed.push_instruction_count if parsed.push_instruction_count else 0.0,
        "total_operand_bytes":float(widths.sum()) if count else 0.0, "mean_operand_bytes":float(widths.mean()) if count else 0.0, "max_operand_bytes":float(widths.max()) if count else 0.0,
        "zero_constant_count":float(zero),"zero_constant_ratio":zero/denominator if count else 0.0,"one_constant_count":float(one),"one_constant_ratio":one/denominator if count else 0.0,
        "small_constant_count":float(small),"small_constant_ratio":small/denominator if count else 0.0,"unique_constant_count":float(len(frequencies)),"unique_constant_ratio":len(frequencies)/denominator if count else 0.0,
        "repeated_constant_count":float(repeated),"repeated_constant_ratio":repeated/denominator if count else 0.0,
        "operand_width_mean":float(widths.mean()) if count else 0.0,"operand_width_median":float(np.median(widths)) if count else 0.0,"operand_width_std":float(widths.std()) if count else 0.0,"operand_width_max":float(widths.max()) if count else 0.0,
    }
    for name,low,high in (("width_1",1,1),("width_2_4",2,4),("width_5_20",5,20),("width_21_32",21,32)):
        amount=bucket(low,high); values[f"{name}_count"]=float(amount); values[f"{name}_ratio"]=amount/denominator if count else 0.0
    for name,width in (("selector_sized_4byte",4),("address_sized_20byte",20),("word_sized_32byte",32)):
        amount=bucket(width,width); values[f"{name}_count"]=float(amount); values[f"{name}_ratio"]=amount/denominator if count else 0.0
    values["repeated_selector_sized_count"]=float(repeated_width(4)); values["repeated_address_sized_count"]=float(repeated_width(20))
    vector=np.asarray([values[name] for name in FEATURE_NAMES],dtype=np.float32)
    if not np.isfinite(vector).all(): raise ValueError("Operand features must be finite")
    return vector


def feature_names_for_groups(groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for group in groups for name in FEATURE_GROUPS[group])


def transform_operand_features(bytecodes: Iterable[object], groups: Sequence[str]=GROUPS, *, small_constant_max:int=255) -> csr_matrix:
    names=feature_names_for_groups(groups); indices=[FEATURE_INDEX[name] for name in names]
    rows=[extract_operand_features(value,small_constant_max=small_constant_max)[indices] for value in bytecodes]
    return csr_matrix(np.vstack(rows) if rows else np.empty((0,len(indices)),dtype=np.float32))
