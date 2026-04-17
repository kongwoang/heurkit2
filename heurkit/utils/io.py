"""
I/O helpers — instance generators for demos and testing.
"""

from __future__ import annotations

import numpy as np


def generate_tsp_coordinates(
    n: int = 20, seed: int | None = None
) -> list[tuple[float, float]]:
    """Return *n* random 2D coordinates in [0, 100]."""
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2)) * 100
    return [(float(x), float(y)) for x, y in coords]


def generate_cvrp_instance(
    n_customers: int = 15,
    capacity: float = 50.0,
    seed: int | None = None,
) -> dict:
    """Return a dict with depot, customers, demands, capacity."""
    rng = np.random.default_rng(seed)
    depot = (50.0, 50.0)
    customers = [
        (float(rng.random() * 100), float(rng.random() * 100))
        for _ in range(n_customers)
    ]
    demands = [float(rng.integers(5, 20)) for _ in range(n_customers)]
    return {
        "depot": depot,
        "customers": customers,
        "demands": demands,
        "capacity": capacity,
    }


def generate_binpacking_items(
    n_items: int = 30,
    capacity: float = 100.0,
    seed: int | None = None,
) -> dict:
    """Return a dict with item_sizes and capacity."""
    rng = np.random.default_rng(seed)
    sizes = [float(rng.integers(10, int(capacity * 0.6))) for _ in range(n_items)]
    return {"item_sizes": sizes, "capacity": capacity}
