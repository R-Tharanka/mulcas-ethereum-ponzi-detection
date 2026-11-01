from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.extension.graph_dataset import (
    FrozenGraphStore,
    aggregate_feature_names,
    chronological_inner_split,
    fit_node_normalizer,
    graph_aggregate_vector,
    normalized_graphs,
)
from src.utils.paths import REPO_ROOT


def test_frozen_store_verifies_and_reconstructs_local_graph_edges() -> None:
    store = FrozenGraphStore(
        REPO_ROOT / "results/extension/phase8r2a/graph_manifest.json",
        REPO_ROOT / "results/extension/phase8r2a/graph_schema.json",
    )
    graphs = store.load_rows([0, 64])
    assert set(graphs) == {0, 64}
    for graph in graphs.values():
        assert graph.x.shape[1] == 55
        assert graph.edge_type.numel() == graph.edge_index.shape[1]
        if graph.edge_index.numel():
            assert int(graph.edge_index.min()) >= 0
            assert int(graph.edge_index.max()) < graph.num_nodes


def test_node_normalizer_fits_only_declared_training_nodes() -> None:
    train = Data(x=torch.tensor([[1.0] * 55, [3.0] * 55]), edge_index=torch.empty((2, 0), dtype=torch.long))
    future = Data(x=torch.tensor([[1000.0] * 55]), edge_index=torch.empty((2, 0), dtype=torch.long))
    normalizer = fit_node_normalizer([train], [7])
    assert normalizer.fitted_row_indices == (7,)
    np.testing.assert_allclose(normalizer.mean, 2.0)
    transformed = normalized_graphs([train, future], normalizer)
    np.testing.assert_allclose(transformed[0].x.numpy().mean(axis=0), 0.0)
    assert float(transformed[1].x.mean()) == 998.0


def test_aggregate_schema_is_fixed_and_finite() -> None:
    graph = Data(
        x=torch.arange(110, dtype=torch.float32).reshape(2, 55),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_type=torch.tensor([0], dtype=torch.long),
    )
    statistics = pd.Series({"graph_density": 0.5, "unresolved_jump_count": 2})
    values = graph_aggregate_vector(graph, statistics)
    assert len(aggregate_feature_names([f"f{i}" for i in range(55)])) == 180
    assert values.shape == (180,)
    assert np.isfinite(values).all()


def test_inner_split_is_strictly_chronological_and_whole_block() -> None:
    frame = pd.DataFrame({
        "row_index": np.arange(20),
        "createdBlockNumber": np.repeat(np.arange(10), 2),
        "label": [0, 1] * 10,
    })
    train, valid, policy = chronological_inner_split(frame, 0.15, minimum_positives=1)
    assert frame.set_index("row_index").loc[train, "createdBlockNumber"].max() < frame.set_index("row_index").loc[valid, "createdBlockNumber"].min()
    assert set(frame.set_index("row_index").loc[train, "createdBlockNumber"]).isdisjoint(
        set(frame.set_index("row_index").loc[valid, "createdBlockNumber"])
    )
    assert "whole_blocks" in policy
