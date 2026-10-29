from __future__ import annotations

import json

import pandas as pd
import yaml

from src.extension.feasibility import (
    DEFAULT_CONFIG_PATH,
    audit_bytecode_values,
    disassemble,
    sample_weight_support,
    summarize_cfg,
    verify_manifest,
)
from src.utils.paths import REPO_ROOT


def test_disassembly_and_cfg_audit_are_deterministic() -> None:
    # PUSH1 0x03; JUMP; JUMPDEST; STOP
    code = "0x6003565b00"
    assert disassemble(code) == disassemble(code)
    first = summarize_cfg(code)
    assert first == summarize_cfg(code)
    assert first.basic_block_count == 2
    assert first.unconditional_jump_edges == 1


def test_dynamic_and_truncated_jumps_are_reported_not_inferred() -> None:
    dynamic = summarize_cfg("0x565b00")
    truncated = summarize_cfg("0x61aa")
    assert dynamic.unresolved_jump_targets == 1
    assert truncated.truncated_pushes == 1
    assert truncated.disassemblable


def test_bytecode_audit_does_not_need_labels() -> None:
    frame = pd.DataFrame({"contractCode": ["0x00", "0x6004565b00"]})
    audit = audit_bytecode_values(frame["contractCode"])
    changed_labels = frame.assign(label=[0, 1])
    assert audit == audit_bytecode_values(changed_labels["contractCode"])


def test_frozen_estimators_expose_sample_weights_without_training() -> None:
    support = sample_weight_support()
    assert all(values["supported"] for values in support.values())


def test_protocol_prohibits_p5_selection_and_model_training() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["temporal_adaptation"]["selection_periods"] == ["P2", "P3", "P4"]
    assert config["tracks"]["track_a"]["p5_selection_use"] == "prohibited"
    assert config["execution"]["train_models"] is False
    assert config["execution"]["download_external_datasets"] is False


def test_frozen_phase7_manifest_and_raw_dataset_are_unchanged() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    result = verify_manifest(REPO_ROOT / config["frozen_manifest"])
    assert result["verified"]
    assert result["artifact_count"] >= 18


def test_generated_feasibility_records_no_training_or_p5_selection() -> None:
    path = REPO_ROOT / "results" / "extension" / "phase8r0" / "feasibility.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["model_training_performed"] is False
    assert result["p5_labels_used_for_selection"] is False
