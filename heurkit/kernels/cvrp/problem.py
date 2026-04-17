"""
CVRP problem definition.

Capacitated Vehicle Routing Problem — simple version.
No time windows, no pickup-delivery.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from heurkit.core.problem import Problem
from heurkit.kernels.cvrp.evaluator import CVRPEvaluator
from heurkit.kernels.cvrp.constructors import GreedySequentialConstructor
from heurkit.kernels.cvrp.neighbors import CVRPNeighborhood


class CVRPProblem(Problem):
    """Simple Capacitated Vehicle Routing Problem.

    Parameters
    ----------
    distance_matrix : NDArray
        (n+1, n+1) distance matrix.  Index 0 is the depot.
    demands : NDArray
        Demand for each node (index 0 = depot, demand must be 0).
    capacity : float
        Vehicle capacity (must be positive).
    coordinates : NDArray or None
        (n+1, 2) coordinates for plotting.
    instance_name : str
        Name for logging.

    Raises
    ------
    ValueError
        If inputs are inconsistent or invalid.
    """

    def __init__(
        self,
        distance_matrix: NDArray,
        demands: NDArray,
        capacity: float,
        coordinates: NDArray | None = None,
        instance_name: str = "CVRP",
    ) -> None:
        distance_matrix = np.asarray(distance_matrix, dtype=float)
        demands = np.asarray(demands, dtype=float)
        if capacity <= 0:
            raise ValueError(f"Vehicle capacity must be positive, got {capacity}")
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(f"Distance matrix must be square, got {distance_matrix.shape}")
        if len(demands) != distance_matrix.shape[0]:
            raise ValueError(
                f"Demands length ({len(demands)}) must match distance matrix "
                f"size ({distance_matrix.shape[0]})"
            )
        if len(demands) < 2:
            raise ValueError("CVRP requires at least 1 customer (demands length >= 2)")
        if any(demands[1:] > capacity):
            raise ValueError("Some customer demands exceed vehicle capacity")

        self.distance_matrix = distance_matrix
        self.demands = demands
        self.capacity = float(capacity)
        self.coordinates = coordinates
        self.instance_name = instance_name
        self.n_customers: int = len(demands) - 1
        self.depot: int = 0

    # ---- factory helpers ---------------------------------------------------

    @classmethod
    def from_coordinates(
        cls,
        depot: tuple[float, float],
        customers: list[tuple[float, float]],
        demands: list[float],
        capacity: float,
        instance_name: str = "CVRP",
    ) -> CVRPProblem:
        """Create CVRP from explicit coordinates."""
        all_coords = np.array([depot] + customers, dtype=float)
        diff = all_coords[:, np.newaxis, :] - all_coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        all_demands = np.array([0.0] + list(demands), dtype=float)
        return cls(dist, all_demands, capacity, all_coords, instance_name)

    @classmethod
    def generate_random(
        cls,
        n_customers: int = 15,
        capacity: float = 50.0,
        seed: int | None = None,
    ) -> CVRPProblem:
        """Generate a random CVRP instance."""
        rng = np.random.default_rng(seed)
        depot = (50.0, 50.0)
        customers = [(float(rng.random() * 100), float(rng.random() * 100))
                      for _ in range(n_customers)]
        max_demand = min(19, int(capacity) - 1)  # ensure demand < capacity
        demands = [float(rng.integers(5, max(6, max_demand + 1))) for _ in range(n_customers)]
        return cls.from_coordinates(
            depot, customers, demands, capacity,
            instance_name=f"CVRP-rand-{n_customers}",
        )

    # ---- Problem interface -------------------------------------------------

    def name(self) -> str:
        return self.instance_name

    def size(self) -> int:
        return self.n_customers

    # ---- kernel defaults ---------------------------------------------------

    def default_evaluator(self) -> CVRPEvaluator:
        return CVRPEvaluator(self)

    def default_constructor(self) -> GreedySequentialConstructor:
        return GreedySequentialConstructor(self)

    def default_neighborhood(self) -> CVRPNeighborhood:
        return CVRPNeighborhood(self)
