from __future__ import annotations

import torch

from src.extension.sequence_model import OpcodeMaskedLanguageModel, masked_opcode_loss
from src.utils.seed import set_random_seed


def test_frozen_sequence_architecture_is_exact() -> None:
    model = OpcodeMaskedLanguageModel(145)
    assert model.maximum_length == 1024
    assert model.embedding.embedding_dim == 128
    assert len(model.encoder.layers) == 2
    layer = model.encoder.layers[0]
    assert layer.self_attn.num_heads == 4
    assert layer.linear1.out_features == 256
    assert layer.dropout.p == 0.1
    assert model.masked_opcode_head.out_features == 145
    assert not any("classif" in name for name, _ in model.named_modules())


def test_positional_encoding_and_masked_loss_are_finite() -> None:
    set_random_seed(42)
    model = OpcodeMaskedLanguageModel(145).eval()
    tokens = torch.zeros((1, 1024), dtype=torch.long)
    masked = torch.zeros_like(tokens, dtype=torch.bool)
    masked[:, ::128] = True
    with torch.no_grad():
        logits = model(tokens)
        loss = masked_opcode_loss(logits, tokens, masked)
    assert logits.shape == (1, 1024, 145)
    assert torch.isfinite(loss)
    assert not torch.equal(model.position.encoding[:, 0], model.position.encoding[:, 1])


def test_repeated_model_setup_is_deterministic() -> None:
    set_random_seed(42)
    torch.manual_seed(42)
    first = OpcodeMaskedLanguageModel(145)
    set_random_seed(42)
    torch.manual_seed(42)
    second = OpcodeMaskedLanguageModel(145)
    for first_value, second_value in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
