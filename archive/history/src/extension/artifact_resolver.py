"""Hash-preserving resolver for historical artifacts relocated with a repository."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.experiments.phase6a import sha256_file
from src.utils.paths import REPO_ROOT


@dataclass(frozen=True)
class ArtifactResolution:
    recorded_path: str
    resolved_path: str
    expected_sha256: str
    relocated: bool
    status: str


def resolve_artifact(recorded: str | Path, expected_sha256: str,
                     approved_roots: tuple[Path, ...] | None = None) -> ArtifactResolution:
    original = Path(recorded)
    candidates = [original]
    roots = approved_roots or (REPO_ROOT / "artifacts", REPO_ROOT / "results" / "artifacts")
    parts = original.parts
    suffix = None
    for marker in ("artifacts", "results"):
        if marker in parts:
            suffix = Path(*parts[parts.index(marker) + 1:])
            for root in roots:
                candidates.append(root / suffix)
            break
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected_sha256:
            return ArtifactResolution(str(original), str(candidate.resolve()), expected_sha256,
                                      candidate != original, "verified")
    raise FileNotFoundError(f"No approved artifact location matches hash {expected_sha256}: {original}")
