"""Default-deny access to any future sealed Track B labels."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_sealed_labels(path: Path, *, final_evaluation: bool = False) -> pd.DataFrame:
    if not final_evaluation:
        raise PermissionError("Track B labels are sealed; explicit final_evaluation=True is required")
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
