"""Frozen Phase 8R2B GIN architecture and deterministic training helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool


class GINClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        if layers not in (2, 3) or hidden_dim not in (64, 128) or dropout not in (0.2, 0.5):
            raise ValueError("Architecture is outside the frozen Phase 8R2B grid.")
        convolutions: list[GINConv] = []
        current = input_dim
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(current, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            convolutions.append(GINConv(mlp, train_eps=False))
            current = hidden_dim
        self.convolutions = nn.ModuleList(convolutions)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for convolution in self.convolutions:
            x = torch.relu(convolution(x, edge_index))
            x = self.dropout(x)
        pooled = torch.cat((global_mean_pool(x, batch), global_max_pool(x, batch)), dim=1)
        return self.head(pooled).view(-1)


def set_torch_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def graph_loader(graphs: Iterable, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(list(graphs), batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def epoch_loss(model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_function(logits, batch.y.view(-1))
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch.num_graphs)
            total_rows += int(batch.num_graphs)
    return total_loss / total_rows


@dataclass(frozen=True)
class TrainingResult:
    model: GINClassifier
    epochs_trained: int
    best_epoch: int
    training_loss: float
    validation_loss: float


def train_with_early_stopping(
    model: GINClassifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    pos_weight: float,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> TrainingResult:
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    last_train = float("nan")
    stale = 0
    for epoch in range(1, max_epochs + 1):
        last_train = epoch_loss(model, train_loader, loss_function, device, optimizer)
        validation = epoch_loss(model, validation_loader, loss_function, device)
        if validation < best_loss - 1e-12:
            best_loss = validation
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Early stopping failed to retain a model state.")
    model.load_state_dict(best_state)
    model.to(device)
    return TrainingResult(model, epoch, best_epoch, last_train, best_loss)


def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            value = batch.row_id
            rows.extend(value.cpu().numpy().astype(int).reshape(-1).tolist() if torch.is_tensor(value) else list(value))
    return np.asarray(rows, dtype=int), np.asarray(probabilities, dtype=float)
