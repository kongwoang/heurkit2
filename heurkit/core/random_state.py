"""
Deterministic random state utility.

Wraps numpy's Generator so that every algorithm and constructor can
share a single seeded RNG for reproducibility.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, PCG64


def make_rng(seed: int | None = None) -> Generator:
    """Create a numpy Generator from an optional seed."""
    return Generator(PCG64(seed))
