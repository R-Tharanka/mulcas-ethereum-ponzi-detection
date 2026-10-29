"""Frozen small opcode Transformer used by the Phase 8R5 benchmark."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, maximum_length: int) -> None:
        super().__init__()
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embedding_dim)
        )
        encoding = torch.zeros(maximum_length, embedding_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1]].to(dtype=values.dtype)


class OpcodeMaskedLanguageModel(nn.Module):
    """Exact frozen Phase 8R5 opcode encoder with no classification head."""

    def __init__(
        self,
        vocabulary_size: int,
        *,
        maximum_length: int = 1024,
        embedding_dim: int = 128,
        encoder_layers: int = 2,
        attention_heads: int = 4,
        feed_forward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.maximum_length = maximum_length
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.position = SinusoidalPositionalEncoding(embedding_dim, maximum_length)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
        self.masked_opcode_head = nn.Linear(embedding_dim, vocabulary_size)

    def forward(
        self, token_ids: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != self.maximum_length:
            raise ValueError(f"Expected [batch, {self.maximum_length}] token IDs.")
        embedded = self.position(self.embedding(token_ids))
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        return self.masked_opcode_head(encoded)


def masked_opcode_loss(
    logits: torch.Tensor, targets: torch.Tensor, masked_positions: torch.Tensor
) -> torch.Tensor:
    if not bool(masked_positions.any()):
        raise ValueError("At least one masked position is required.")
    return nn.functional.cross_entropy(logits[masked_positions], targets[masked_positions])
