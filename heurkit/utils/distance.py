"""
Distance utilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def euclidean_distance_matrix(coords: NDArray) -> NDArray:
    """Compute a full Euclidean distance matrix from (n, 2) coordinates."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        diff = coords[i] - coords
        dist[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    return dist
