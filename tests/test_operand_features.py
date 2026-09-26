"""Deterministic EVM operand parsing and normalized-feature tests."""

from __future__ import annotations

import numpy as np

from src.features.operands import FEATURE_INDEX, FEATURE_NAMES, disassemble_operands, extract_operand_features, transform_operand_features

def _value(vector:np.ndarray,name:str)->float: return float(vector[FEATURE_INDEX[name]])

def test_push1_through_push32_widths_and_determinism()->None:
    code=b"".join(bytes([0x5F+width])+bytes([width])*width for width in range(1,33)); first=disassemble_operands("0x"+code.hex()); second=disassemble_operands("0x"+code.hex())
    assert first==second and [len(x) for x in first.operands]==list(range(1,33))

def test_truncated_push_is_explicit_and_not_inferred()->None:
    parsed=disassemble_operands("0x63aabb")
    assert parsed.push_instruction_count==1 and parsed.malformed_truncated_push_count==1 and parsed.operands==()

def test_schema_has_no_raw_operand_identities()->None:
    assert len(FEATURE_NAMES)==37 and len(set(FEATURE_NAMES))==37
    assert not any("0x" in name or "exact" in name or "address_identity" in name or "selector_identity" in name for name in FEATURE_NAMES)

def test_width_pattern_indicators()->None:
    code=bytes([0x63])+b"abcd"+bytes([0x73])+bytes(range(20))+bytes([0x7F])+bytes(range(32)); vector=extract_operand_features("0x"+code.hex())
    assert _value(vector,"selector_sized_4byte_count")==1 and _value(vector,"address_sized_20byte_count")==1 and _value(vector,"word_sized_32byte_count")==1
    assert _value(vector,"selector_sized_4byte_ratio")==_value(vector,"address_sized_20byte_ratio")==_value(vector,"word_sized_32byte_ratio")==np.float32(1/3)

def test_zero_small_unique_and_repeated_metrics()->None:
    code=bytes([0x60,0,0x60,1,0x60,1,0x61,1,0]); vector=extract_operand_features("0x"+code.hex(),small_constant_max=255)
    assert _value(vector,"zero_constant_count")==1 and _value(vector,"one_constant_count")==2
    assert _value(vector,"small_constant_count")==3 and _value(vector,"unique_constant_count")==3
    assert _value(vector,"repeated_constant_count")==1 and _value(vector,"repeated_constant_ratio")==0.25

def test_empty_null_malformed_and_unknown_are_finite()->None:
    for value in (None,"","0x","not-hex","0x0c"):
        assert np.isfinite(extract_operand_features(value)).all()

def test_full_runtime_metadata_like_tail_is_preserved()->None:
    vector=extract_operand_features("0x6001a164627a7a72305820")
    assert _value(vector,"parseable_operand_count")>=2

def test_labels_and_other_contracts_cannot_affect_local_vector()->None:
    target="0x60016002"; before=transform_operand_features([target,"0x6003"])[0].toarray(); after=transform_operand_features([target,"0x7f"+"ff"*32])[0].toarray()
    assert np.array_equal(before,after)

