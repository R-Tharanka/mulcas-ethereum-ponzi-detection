"""Repository-wide environment gates for genuinely CUDA-dependent tests."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if cuda_available:
        return
    cuda_tests = {
        "tests/test_phase8r5b.py::test_cuda_gate_matches_approved_environment",
        "tests/test_sequence_embeddings.py::test_cuda_extraction_returns_one_finite_128d_embedding_per_contract",
    }
    marker = pytest.mark.skip(reason="CUDA is unavailable in this environment; historical CUDA results are unchanged")
    for item in items:
        normalized = item.nodeid.replace("\\", "/")
        if normalized in cuda_tests:
            item.add_marker(marker)
