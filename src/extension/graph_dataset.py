"""Verified readers and fixed transforms for the frozen Phase 8R2A corpus."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils.paths import REPO_ROOT


EXPECTED_CORPUS_SHA256 = "ece393232abad38fb329f4ccbe12e617d9c3ec90ba5f265d867ccf5670fec680"
EXPECTED_SCHEMA_SHA256 = "13cf8f4be61e83da99f31649de57f7e65d2d8e2a7cba960ba4f24dbae3e69ec0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NodeNormalizer:
    mean: np.ndarray
    std: np.ndarray
    fitted_row_indices: tuple[int, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


class FrozenGraphStore:
    """Read frozen compressed chunks and return independent PyG graphs."""

    def __init__(self, manifest_path: Path, schema_path: Path, verify_chunks: bool = True) -> None:
        self.manifest_path = manifest_path
        self.schema_path = schema_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if self.manifest["corpus_sha256"] != EXPECTED_CORPUS_SHA256:
            raise RuntimeError("Frozen graph corpus hash does not match Phase 8R2A.")
        if self.manifest["schema_sha256"] != EXPECTED_SCHEMA_SHA256:
            raise RuntimeError("Frozen graph schema hash does not match Phase 8R2A.")
        if self.manifest["graph_count"] != 6498 or self.schema["node_feature_dimension"] != 55:
            raise RuntimeError("Frozen graph dimensions changed.")
        self._locations: dict[int, tuple[dict[str, object], int]] = {}
        for artifact in self.manifest["artifacts"]:
            path = REPO_ROOT / str(artifact["path"])
            if not path.is_file() or (verify_chunks and sha256_file(path) != artifact["sha256"]):
                raise RuntimeError(f"Frozen graph artifact failed verification: {path}")
            first = int(artifact["first_row_index"])
            for local in range(int(artifact["graph_count"])):
                self._locations[first + local] = (artifact, local)

    def load_rows(self, row_indices: Iterable[int], labels: dict[int, int] | None = None) -> dict[int, Data]:
        requested = sorted({int(value) for value in row_indices})
        grouped: dict[str, list[tuple[int, int]]] = {}
        for row_index in requested:
            if row_index not in self._locations:
                raise KeyError(f"Graph row {row_index} is absent from frozen corpus.")
            artifact, local = self._locations[row_index]
            grouped.setdefault(str(artifact["path"]), []).append((row_index, local))
        output: dict[int, Data] = {}
        for relative, members in grouped.items():
            with np.load(REPO_ROOT / relative, allow_pickle=False) as arrays:
                for row_index, local in members:
                    node_start, node_end = int(arrays["node_ptr"][local]), int(arrays["node_ptr"][local + 1])
                    edge_start, edge_end = int(arrays["edge_ptr"][local]), int(arrays["edge_ptr"][local + 1])
                    x = np.asarray(arrays["x"][node_start:node_end], dtype=np.float32)
                    edges = np.asarray(arrays["edge_index"][:, edge_start:edge_end], dtype=np.int64) - node_start
                    kwargs: dict[str, torch.Tensor | int] = {
                        "x": torch.from_numpy(x.copy()),
                        "edge_index": torch.from_numpy(edges.copy()),
                        "edge_type": torch.from_numpy(np.asarray(arrays["edge_type"][edge_start:edge_end], dtype=np.int64).copy()),
                        # PyG increments attributes containing "index" during batching.
                        "row_id": row_index,
                    }
                    if labels is not None:
                        kwargs["y"] = torch.tensor([float(labels[row_index])], dtype=torch.float32)
                    output[row_index] = Data(**kwargs)
        return output


def fit_node_normalizer(graphs: Iterable[Data], row_indices: Iterable[int]) -> NodeNormalizer:
    """Fit population moments using nodes from declared training graphs only."""
    rows = tuple(int(value) for value in row_indices)
    graph_list = list(graphs)
    if not graph_list:
        raise ValueError("At least one training graph is required.")
    count = 0
    total = np.zeros(55, dtype=np.float64)
    squared = np.zeros(55, dtype=np.float64)
    for graph in graph_list:
        values = graph.x.detach().cpu().numpy().astype(np.float64)
        count += len(values)
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std == 0.0] = 1.0
    return NodeNormalizer(mean.astype(np.float32), std.astype(np.float32), rows)


def normalized_graphs(graphs: Iterable[Data], normalizer: NodeNormalizer) -> list[Data]:
    output: list[Data] = []
    for graph in graphs:
        clone = graph.clone()
        clone.x = torch.from_numpy(normalizer.transform(graph.x.detach().cpu().numpy()))
        output.append(clone)
    return output


def aggregate_feature_names(node_names: list[str]) -> list[str]:
    topology = [
        "node_count", "edge_count", "graph_density", "mean_in_degree", "mean_out_degree",
        "max_in_degree", "max_out_degree", "zero_outdegree_node_count", "fall_through_edge_count",
        "static_jump_edge_count", "conditional_jump_edge_count", "fall_through_edge_ratio",
        "static_jump_edge_ratio", "conditional_jump_edge_ratio", "unresolved_jump_count",
    ]
    return topology + [f"node_mean__{name}" for name in node_names] + [
        f"node_max__{name}" for name in node_names
    ] + [f"node_std__{name}" for name in node_names]


def graph_aggregate_vector(graph: Data, statistics: pd.Series) -> np.ndarray:
    x = graph.x.detach().cpu().numpy()
    node_count = len(x)
    edge_count = int(graph.edge_index.shape[1])
    indegree = np.bincount(graph.edge_index[1].numpy(), minlength=node_count)
    outdegree = np.bincount(graph.edge_index[0].numpy(), minlength=node_count)
    edge_types = np.bincount(graph.edge_type.numpy(), minlength=3)
    denominator = float(edge_count) if edge_count else 1.0
    topology = np.asarray([
        node_count, edge_count, float(statistics["graph_density"]), indegree.mean(), outdegree.mean(),
        indegree.max(initial=0), outdegree.max(initial=0), int((outdegree == 0).sum()),
        *edge_types.tolist(), *(edge_types / denominator).tolist(), int(statistics["unresolved_jump_count"]),
    ], dtype=np.float64)
    aggregate = np.concatenate((topology, x.mean(axis=0), x.max(axis=0), x.std(axis=0)))
    if not np.isfinite(aggregate).all():
        raise ValueError("Aggregate graph features must be finite.")
    return aggregate


def chronological_inner_split(frame: pd.DataFrame, fraction: float = 0.15, minimum_positives: int = 5) -> tuple[np.ndarray, np.ndarray, str]:
    """Split historical rows at a whole-block boundary without future input."""
    ordered = frame.sort_values(["createdBlockNumber", "row_index"], kind="mergesort")
    target = max(1, int(np.ceil(len(ordered) * fraction)))
    blocks = ordered["createdBlockNumber"].drop_duplicates().to_numpy()
    candidates = [
        int(value) for value in blocks
        if int((ordered["createdBlockNumber"] >= value).sum()) >= target
    ]
    boundary = max(candidates)
    inner_train = ordered[ordered["createdBlockNumber"] < boundary]
    inner_valid = ordered[ordered["createdBlockNumber"] >= boundary]
    policy = "latest_15_percent_whole_blocks"
    if inner_valid["label"].sum() < minimum_positives:
        positives = ordered.loc[ordered["label"].eq(1), "createdBlockNumber"].drop_duplicates().to_numpy()
        if len(positives) < minimum_positives + 1:
            raise ValueError("Historical data cannot support chronological inner validation.")
        boundary = int(positives[-minimum_positives])
        inner_train = ordered[ordered["createdBlockNumber"] < boundary]
        inner_valid = ordered[ordered["createdBlockNumber"] >= boundary]
        policy = f"latest_{minimum_positives}_positive_occurrences_whole_blocks"
    if inner_train.empty or inner_valid.empty or inner_train["label"].nunique() != 2:
        raise ValueError("Chronological inner split is unusable.")
    if inner_train["createdBlockNumber"].max() >= inner_valid["createdBlockNumber"].min():
        raise AssertionError("Inner validation must be strictly later than inner training.")
    return inner_train["row_index"].to_numpy(), inner_valid["row_index"].to_numpy(), policy
