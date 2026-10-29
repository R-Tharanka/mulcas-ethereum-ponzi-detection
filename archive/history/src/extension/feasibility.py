"""Deterministic Phase 8R0 feasibility audits without model training."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.features.operands import OPCODE_NAMES, decode_bytecode
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "extension" / "phase8r0.yaml"
TERMINATORS = {"STOP", "RETURN", "REVERT", "INVALID", "SELFDESTRUCT"}
BRANCHES = {"JUMP", "JUMPI"}


@dataclass(frozen=True)
class Instruction:
    """One decoded EVM instruction and its byte offset."""

    pc: int
    opcode: int
    name: str
    operand: bytes
    truncated: bool = False


@dataclass(frozen=True)
class Disassembly:
    """Deterministic bytecode disassembly result."""

    instructions: tuple[Instruction, ...]
    invalid_hex: bool
    empty: bool
    truncated_pushes: int
    unsupported_opcodes: int


@dataclass(frozen=True)
class CFGSummary:
    """Compact control-flow graph feasibility summary for one contract."""

    disassemblable: bool
    instruction_count: int
    basic_block_count: int
    fallthrough_edges: int
    unconditional_jump_edges: int
    conditional_jump_edges: int
    unresolved_jump_targets: int
    truncated_pushes: int
    unsupported_opcodes: int
    invalid_hex: bool
    empty: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disassemble(bytecode: object) -> Disassembly:
    """Decode bytecode while retaining malformed PUSH instructions explicitly."""
    code = decode_bytecode(bytecode)
    if code is None:
        return Disassembly((), True, False, 0, 0)
    if not code:
        return Disassembly((), False, True, 0, 0)

    instructions: list[Instruction] = []
    pc = 0
    truncated = 0
    unsupported = 0
    while pc < len(code):
        start = pc
        opcode = code[pc]
        pc += 1
        name = OPCODE_NAMES.get(opcode, "INVALID")
        if opcode not in OPCODE_NAMES:
            unsupported += 1
        width = opcode - 0x5F if 0x60 <= opcode <= 0x7F else 0
        available = min(width, len(code) - pc)
        operand = code[pc : pc + available]
        is_truncated = available != width
        if is_truncated:
            truncated += 1
        instructions.append(Instruction(start, opcode, name, operand, is_truncated))
        pc += available
    return Disassembly(tuple(instructions), False, False, truncated, unsupported)


def _static_jump_target(block: tuple[Instruction, ...], valid_destinations: set[int]) -> int | None:
    """Resolve only the conservative PUSH-immediately-before-JUMP pattern."""
    if len(block) < 2:
        return None
    candidate = block[-2]
    if not candidate.name.startswith("PUSH") or candidate.truncated:
        return None
    target = int.from_bytes(candidate.operand, "big")
    return target if target in valid_destinations else None


def summarize_cfg(bytecode: object) -> CFGSummary:
    """Build conservative basic blocks and count statically identifiable edges."""
    parsed = disassemble(bytecode)
    instructions = parsed.instructions
    if not instructions:
        return CFGSummary(
            False, 0, 0, 0, 0, 0, 0, parsed.truncated_pushes,
            parsed.unsupported_opcodes, parsed.invalid_hex, parsed.empty,
        )

    leaders = {instructions[0].pc}
    for index, instruction in enumerate(instructions):
        if instruction.name == "JUMPDEST":
            leaders.add(instruction.pc)
        if instruction.name in TERMINATORS | BRANCHES and index + 1 < len(instructions):
            leaders.add(instructions[index + 1].pc)

    blocks: list[tuple[Instruction, ...]] = []
    current: list[Instruction] = []
    for instruction in instructions:
        if current and instruction.pc in leaders:
            blocks.append(tuple(current))
            current = []
        current.append(instruction)
    if current:
        blocks.append(tuple(current))

    destinations = {item.pc for item in instructions if item.name == "JUMPDEST"}
    fallthrough = 0
    unconditional = 0
    conditional = 0
    unresolved = 0
    for index, block in enumerate(blocks):
        final = block[-1]
        has_next = index + 1 < len(blocks)
        if final.name == "JUMP":
            if _static_jump_target(block, destinations) is None:
                unresolved += 1
            else:
                unconditional += 1
        elif final.name == "JUMPI":
            if has_next:
                fallthrough += 1
            if _static_jump_target(block, destinations) is None:
                unresolved += 1
            else:
                conditional += 1
        elif final.name not in TERMINATORS and has_next:
            fallthrough += 1

    return CFGSummary(
        True,
        len(instructions),
        len(blocks),
        fallthrough,
        unconditional,
        conditional,
        unresolved,
        parsed.truncated_pushes,
        parsed.unsupported_opcodes,
        parsed.invalid_hex,
        parsed.empty,
    )


def _distribution(values: Iterable[int]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"min": 0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
    }


def audit_bytecode_values(values: Iterable[object]) -> dict[str, Any]:
    """Aggregate CFG feasibility without requiring labels or other contracts."""
    summaries = [summarize_cfg(value) for value in values]
    return {
        "contracts_audited": len(summaries),
        "disassemblable_contracts": sum(item.disassemblable for item in summaries),
        "invalid_hex_contracts": sum(item.invalid_hex for item in summaries),
        "empty_contracts": sum(item.empty for item in summaries),
        "contracts_with_truncated_pushes": sum(item.truncated_pushes > 0 for item in summaries),
        "truncated_push_count": sum(item.truncated_pushes for item in summaries),
        "contracts_with_unsupported_opcodes": sum(item.unsupported_opcodes > 0 for item in summaries),
        "unsupported_opcode_count": sum(item.unsupported_opcodes for item in summaries),
        "graphs_with_basic_blocks": sum(item.basic_block_count > 0 for item in summaries),
        "graphs_with_fallthrough_edges": sum(item.fallthrough_edges > 0 for item in summaries),
        "graphs_with_static_unconditional_jump_edges": sum(item.unconditional_jump_edges > 0 for item in summaries),
        "graphs_with_static_conditional_jump_edges": sum(item.conditional_jump_edges > 0 for item in summaries),
        "graphs_with_unresolved_jump_targets": sum(item.unresolved_jump_targets > 0 for item in summaries),
        "edge_totals": {
            "fallthrough": sum(item.fallthrough_edges for item in summaries),
            "static_unconditional_jump": sum(item.unconditional_jump_edges for item in summaries),
            "static_conditional_jump": sum(item.conditional_jump_edges for item in summaries),
            "unresolved_jump_targets": sum(item.unresolved_jump_targets for item in summaries),
        },
        "instruction_count": _distribution(item.instruction_count for item in summaries),
        "basic_block_count": _distribution(item.basic_block_count for item in summaries),
        "identified_edge_count": _distribution(
            item.fallthrough_edges + item.unconditional_jump_edges + item.conditional_jump_edges
            for item in summaries
        ),
    }


def audit_dataset_bytecode(dataset: Path) -> dict[str, Any]:
    """Audit runtime and creation bytecode fields without loading labels."""
    frame = pd.read_csv(dataset, usecols=["contractCode", "creationCode"])
    return {
        "contract_count": int(len(frame)),
        "fields_loaded": ["contractCode", "creationCode"],
        "labels_loaded": False,
        "runtime": audit_bytecode_values(frame["contractCode"]),
        "creation": audit_bytecode_values(frame["creationCode"]),
        "metadata_policy": "preserve full bytecode, matching the frozen reproduction policy",
        "jump_resolution_policy": (
            "resolve only PUSH-immediately-before-JUMP/JUMPI targets that point to JUMPDEST; "
            "record all other dynamic targets as unresolved"
        ),
    }


def sample_weight_support() -> dict[str, Any]:
    """Inspect fit signatures only; no estimator is fitted."""
    families = {
        "ridge": RidgeClassifier,
        "svm": SVC,
        "xgboost": XGBClassifier,
    }
    return {
        name: {
            "supported": "sample_weight" in inspect.signature(model.fit).parameters,
            "fit_signature": str(inspect.signature(model.fit)),
        }
        for name, model in families.items()
    }


def verify_manifest(path: Path) -> dict[str, Any]:
    expected = yaml.safe_load(path.read_text(encoding="utf-8"))["sha256"]
    observed: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        target = REPO_ROOT / relative
        actual = sha256_file(target)
        if actual != expected_hash:
            raise RuntimeError(f"Frozen artifact changed: {relative}")
        observed[relative] = actual
    return {"verified": True, "artifact_count": len(observed), "sha256": observed}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = _resolve(config["dataset"])
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = verify_manifest(_resolve(config["frozen_manifest"]))
    graph_audit = audit_dataset_bytecode(dataset)
    graph_audit.update({
        "authoritative_runtime_source": "contractCode",
        "creation_source": "creationCode",
        "raw_dataset_sha256": sha256_file(dataset),
        "deterministic": True,
        "model_training_performed": False,
    })
    external = {
        "audit_date": config["external_dataset_audit"]["audit_date"],
        "large_datasets_downloaded": False,
        "candidates": config["external_dataset_audit"]["candidates"],
        "required_overlap_audit": config["external_dataset_audit"]["required_overlap_audit"],
    }
    feasibility = {
        "phase": "8R0",
        "research_question": config["research_question"],
        "hypotheses": config["hypotheses"],
        "tracks": config["tracks"],
        "published_reference": config["published_reference"],
        "reference_reconciliation": config["reference_reconciliation"],
        "p5_benchmark_interpretation": config["p5_benchmark_interpretation"],
        "temporal_adaptation": {
            "sample_weight_support": sample_weight_support(),
            "candidate_strategies": config["temporal_adaptation"]["candidate_strategies"],
            "selection_periods": config["temporal_adaptation"]["selection_periods"],
            "selection_rule": config["temporal_adaptation"]["selection_rule"],
            "p5_selection_use": "prohibited",
        },
        "graph_model_plan": config["graph_model_plan"],
        "novelty_routing_plan": config["novelty_routing_plan"],
        "dependencies": config["dependencies"],
        "frozen_artifacts": frozen,
        "raw_dataset_sha256": sha256_file(dataset),
        "model_training_performed": False,
        "p5_labels_used_for_selection": False,
        "deterministic": True,
    }

    outputs = {
        "feasibility": output_dir / "feasibility.json",
        "graph_audit": output_dir / "graph_audit.json",
        "external": output_dir / "external_dataset_candidates.json",
    }
    outputs["feasibility"].write_text(json.dumps(feasibility, indent=2), encoding="utf-8")
    outputs["graph_audit"].write_text(json.dumps(graph_audit, indent=2), encoding="utf-8")
    outputs["external"].write_text(json.dumps(external, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    for name, path in run(args.config).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
