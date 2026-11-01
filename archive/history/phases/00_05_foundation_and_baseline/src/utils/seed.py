"""Deterministic random seed helpers for reproducible experiments."""

from __future__ import annotations

import os
import random
from typing import Final


DEFAULT_SEED: Final[int] = 42


def set_random_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed supported random number generators and return the seed used.

    Set ``PYTHONHASHSEED`` before starting Python for fully deterministic hash
    randomization. This helper records the requested value for child processes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        return seed

    np.random.seed(seed)
    return seed
