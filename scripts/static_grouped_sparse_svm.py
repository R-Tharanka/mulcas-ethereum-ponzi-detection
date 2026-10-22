"""Run the fixed clone-capped grouped sparse-SVM diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.static_grouped_sparse_svm import run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

