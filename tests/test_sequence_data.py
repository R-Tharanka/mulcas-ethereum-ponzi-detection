from __future__ import annotations

from src.extension.sequence_data import (
    SPECIAL_TOKENS,
    build_vocabulary,
    canonical_opcode_tokens,
    fixed_size_chunks,
    padded_ids,
    prefix_suffix_truncate,
    prefix_truncate,
)


def test_opcode_tokens_are_canonical_deterministic_and_operand_free() -> None:
    document = "PUSH1 0x80 PUSH4 0xdeadbeef MSTORE STOP"
    expected = ("PUSH1", "PUSH4", "MSTORE", "STOP")
    assert canonical_opcode_tokens(document) == expected
    assert canonical_opcode_tokens(document) == expected
    assert all(not token.startswith("0x") for token in expected)


def test_vocabulary_is_sorted_and_fit_only_from_supplied_history() -> None:
    history = [("PUSH1", "STOP"), ("ADD", "PUSH1")]
    vocabulary = build_vocabulary(history)
    before = dict(vocabulary.token_to_id)
    assert tuple(before) == (*SPECIAL_TOKENS, "ADD", "PUSH1", "STOP")
    assert vocabulary.encode(("FUTURE_ONLY",)) == [vocabulary.unk_id]
    assert vocabulary.token_to_id == before


def test_truncation_and_chunking_are_deterministic() -> None:
    tokens = tuple(f"OP{index}" for index in range(9))
    assert prefix_truncate(tokens, 5) == tokens[:5]
    assert prefix_suffix_truncate(tokens, 5) == (*tokens[:3], *tokens[-2:])
    assert fixed_size_chunks(tokens, 4) == (tokens[:4], tokens[4:8], tokens[8:])
    assert fixed_size_chunks((), 4) == ((),)


def test_padding_uses_only_fixed_special_tokens() -> None:
    vocabulary = build_vocabulary([("ADD", "STOP")])
    ids, mask = padded_ids(("ADD",), vocabulary, 3)
    assert ids == [vocabulary.token_to_id["ADD"], vocabulary.pad_id, vocabulary.pad_id]
    assert mask == [False, True, True]

