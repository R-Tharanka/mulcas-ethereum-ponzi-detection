"""Verify frozen manifests while honoring explicit successor test migrations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import REPO_ROOT


MIGRATION_RECORD = REPO_ROOT / "results/extension/phase8r6d/test_anchor_migration.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_test_migrations(path: Path = MIGRATION_RECORD) -> dict[str, dict[str, Any]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("authorized_by_successor_phase") != "phase8r6d":
        raise RuntimeError("Test migration lacks Phase 8R6D successor authorization.")
    if record.get("scope") != "successor_test_assertions_only":
        raise RuntimeError("Test migration scope is not sufficiently narrow.")
    migrations: dict[str, dict[str, Any]] = {}
    for item in record.get("changes", []):
        relative = str(item["test_file"]).replace("\\", "/")
        if relative not in {"tests/test_phase8r6b.py", "tests/test_phase8r6c.py"}:
            raise RuntimeError(f"Unauthorized successor migration path: {relative}")
        migrations[relative] = item
    return migrations


def verify_manifest_with_successor_migrations(
    manifest_path: Path,
    *,
    migration_path: Path = MIGRATION_RECORD,
) -> dict[str, str]:
    """Verify exact hashes, allowing only recorded old-to-new test migrations."""
    expected = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["sha256"]
    migrations = load_test_migrations(migration_path)
    observed: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        normalized = str(relative).replace("\\", "/")
        target = REPO_ROOT / relative
        if not target.is_file():
            raise FileNotFoundError(f"Frozen file is missing: {relative}")
        actual = _sha256(target)
        if actual != expected_hash:
            migration = migrations.get(normalized)
            if not (
                migration
                and migration["old_sha256"] == expected_hash
                and migration["new_sha256"] == actual
            ):
                raise RuntimeError(f"Frozen artifact hash mismatch: {relative}")
        observed[normalized] = actual
    return observed
