"""Command-line entry point for Phase 9A."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.phase9.protocol_gap import DEFAULT_CONFIG_PATH, load_config, run_phase9a


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 9A protocol-gap decomposition.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    results = run_phase9a(load_config(args.config))
    print(f"Phase 9A status: {results['status']}")
    print("No new model family was trained and no threshold was changed.")


if __name__ == "__main__":
    main()
