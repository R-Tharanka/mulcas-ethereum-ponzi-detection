"""Run the duplicate-group-safe successor to the 5ed8509 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.static_grouped_mulcas import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-audit-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=20_000)
    args = parser.parse_args()
    result = run(
        seed=args.seed,
        workers=args.workers,
        max_features=args.max_features,
        split_audit_only=args.split_audit_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
