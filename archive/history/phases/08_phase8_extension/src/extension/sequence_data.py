"""Deterministic opcode-sequence preparation for temporal experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.features.structural import tokenize_opcode_names


SPECIAL_TOKENS: tuple[str, ...] = ("PAD", "MASK", "UNK")


@dataclass(frozen=True)
class OpcodeVocabulary:
    token_to_id: dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["PAD"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["MASK"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["UNK"]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]


def canonical_opcode_tokens(document: object) -> tuple[str, ...]:
    """Return canonical names while discarding hexadecimal operand literals."""
    return tuple(tokenize_opcode_names(document))


def build_vocabulary(sequences: Iterable[Sequence[str]]) -> OpcodeVocabulary:
    """Fit a deterministic vocabulary from the supplied historical corpus only."""
    observed = sorted({token for sequence in sequences for token in sequence if token not in SPECIAL_TOKENS})
    ordered = (*SPECIAL_TOKENS, *observed)
    return OpcodeVocabulary({token: index for index, token in enumerate(ordered)})


def prefix_truncate(tokens: Sequence[str], maximum_length: int) -> tuple[str, ...]:
    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    return tuple(tokens[:maximum_length])


def prefix_suffix_truncate(tokens: Sequence[str], maximum_length: int) -> tuple[str, ...]:
    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    if len(tokens) <= maximum_length:
        return tuple(tokens)
    prefix_length = (maximum_length + 1) // 2
    suffix_length = maximum_length - prefix_length
    return tuple(tokens[:prefix_length]) + tuple(tokens[-suffix_length:]) if suffix_length else tuple(tokens[:prefix_length])


def fixed_size_chunks(tokens: Sequence[str], chunk_length: int) -> tuple[tuple[str, ...], ...]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if not tokens:
        return ((),)
    return tuple(tuple(tokens[start : start + chunk_length]) for start in range(0, len(tokens), chunk_length))


def padded_ids(
    tokens: Sequence[str], vocabulary: OpcodeVocabulary, maximum_length: int
) -> tuple[list[int], list[bool]]:
    selected = prefix_suffix_truncate(tokens, maximum_length)
    ids = vocabulary.encode(selected)
    padding = maximum_length - len(ids)
    return ids + [vocabulary.pad_id] * padding, [False] * len(ids) + [True] * padding
