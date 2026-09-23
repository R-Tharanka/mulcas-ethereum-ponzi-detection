"""Repository-relative filesystem paths."""

from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = REPO_ROOT / "configs"
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DOCS_DIR: Path = REPO_ROOT / "docs"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
RESULTS_DIR: Path = REPO_ROOT / "results"
SRC_DIR: Path = REPO_ROOT / "src"
TESTS_DIR: Path = REPO_ROOT / "tests"


def project_path(*parts: str | Path) -> Path:
    """Return a path resolved relative to the repository root."""
    return REPO_ROOT.joinpath(*parts)
