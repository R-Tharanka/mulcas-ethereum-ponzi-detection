from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from src.extension.feasibility import verify_manifest
from src.experiments.phase6a import sha256_file
from src.extension.phase8r2a import (
    DEFAULT_CONFIG_PATH,
    chunk_arrays,
    graph_content_hash,
    load_config,
    write_deterministic_npz,
)
from src.extension.cfg import build_cfg
from src.utils.paths import REPO_ROOT


def test_all_frozen_manifests_and_raw_dataset_are_unchanged() -> None:
    config = load_config()
    assert verify_manifest(config["frozen_phase7_manifest"])["verified"]
    assert verify_manifest(config["frozen_phase8r0_manifest"])["verified"]
    assert verify_manifest(config["frozen_phase8r1a_manifest"])["verified"]


def test_phase_is_construction_only_and_uses_runtime_bytecode() -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["source"]["bytecode_column"] == "contractCode"
    assert raw["source"]["creation_code_primary_view"] is False
    assert not any(raw["execution"].values())
    assert raw["jump_resolution"]["unresolved_fanout"] is False


def test_chunk_format_and_writer_are_byte_deterministic(tmp_path: Path) -> None:
    cfg = build_cfg("0x6003565b00")
    arrays = chunk_arrays([(7, cfg)])
    assert set(("x", "edge_index", "edge_type", "row_index", "node_ptr", "edge_ptr")).issubset(arrays)
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    assert graph_content_hash(cfg) == graph_content_hash(build_cfg("0x6003565b00"))


def test_generated_corpus_covers_all_contracts_without_models() -> None:
    output = REPO_ROOT / "results/extension/phase8r2a"
    results = json.loads((output / "phase8r2a_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "graph_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((output / "graph_schema.json").read_text(encoding="utf-8"))
    assert results["aggregate_graph_audit"]["graphs_created"] == 6498
    assert results["aggregate_graph_audit"]["non_finite_feature_count"] == 0
    assert results["labels_loaded_for_graph_construction"] is False
    assert results["classifier_trained"] is False and results["gnn_trained"] is False
    assert results["p5_model_evaluation_performed"] is False
    assert manifest["graph_count"] == 6498
    assert manifest["labels_in_graph_artifacts"] is False
    assert schema["node_feature_dimension"] == 55


def test_generated_chunk_hashes_and_shapes_match_manifest() -> None:
    manifest = json.loads(
        (REPO_ROOT / "results/extension/phase8r2a/graph_manifest.json").read_text(encoding="utf-8")
    )
    assert sum(item["graph_count"] for item in manifest["artifacts"]) == 6498
    for item in (manifest["artifacts"][0], manifest["artifacts"][-1]):
        path = REPO_ROOT / item["path"]
        assert path.is_file()
        assert sha256_file(path) == item["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            assert arrays["x"].shape[1] == 55
            assert arrays["edge_index"].shape[0] == 2
            assert len(arrays["row_index"]) == item["graph_count"]


def test_no_torch_dependency_is_introduced() -> None:
    sources = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in ("src/extension/cfg.py", "src/extension/graph_features.py", "src/extension/phase8r2a.py")
    )
    assert "import torch" not in sources
    assert "torch_geometric" not in sources
