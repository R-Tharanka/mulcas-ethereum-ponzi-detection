from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from src.extension.graph_models import (
    GINClassifier,
    graph_loader,
    predict_probabilities,
    set_torch_seed,
    train_with_early_stopping,
)


def _graphs() -> list[Data]:
    return [
        Data(x=torch.ones((2, 55)), edge_index=torch.tensor([[0], [1]]), y=torch.tensor([0.0]), row_id=1),
        Data(x=torch.ones((5, 55)) * 2, edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), y=torch.tensor([1.0]), row_id=2),
    ]


def test_gin_forward_supports_variable_graph_sizes() -> None:
    model = GINClassifier(55, 64, 2, 0.2)
    batch = next(iter(graph_loader(_graphs(), 2, False, 7)))
    assert model(batch.x, batch.edge_index, batch.batch).shape == (2,)
    assert batch.row_id.tolist() == [1, 2]


def test_architecture_rejects_values_outside_frozen_grid() -> None:
    try:
        GINClassifier(55, 32, 2, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-frozen architecture was accepted.")


def test_tiny_training_is_deterministic_and_uses_fixed_pos_weight() -> None:
    outputs = []
    for _ in range(2):
        set_torch_seed(42)
        model = GINClassifier(55, 64, 2, 0.2)
        train = graph_loader(_graphs(), 2, True, 42)
        valid = graph_loader(_graphs(), 2, False, 42)
        fitted = train_with_early_stopping(model, train, valid, 1.0, 0.001, 0.0001, 2, 2, torch.device("cpu"))
        rows, probabilities = predict_probabilities(fitted.model, valid, torch.device("cpu"))
        outputs.append((rows, probabilities, fitted.best_epoch, fitted.validation_loss))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_allclose(outputs[0][1], outputs[1][1])
    assert outputs[0][2:] == outputs[1][2:]


def test_loss_policy_is_bce_with_logits() -> None:
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    assert torch.isfinite(loss(torch.tensor([0.0]), torch.tensor([1.0])))
