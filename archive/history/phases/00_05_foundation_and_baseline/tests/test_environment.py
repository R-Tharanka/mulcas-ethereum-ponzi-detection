"""Minimal import and reproducibility smoke tests."""

from __future__ import annotations

import random

from src.utils.paths import REPO_ROOT, project_path
from src.utils.seed import DEFAULT_SEED, set_random_seed


def test_repository_paths_resolve_from_project_root() -> None:
    assert project_path("README.md").is_file()
    assert project_path("src").is_dir()


def test_seed_helper_is_deterministic_for_python_random() -> None:
    set_random_seed(DEFAULT_SEED)
    first = random.random()

    set_random_seed(DEFAULT_SEED)
    second = random.random()

    assert first == second
