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
        Demand for each customer (index 0 = depot, demand = 0).
    capacity : float
        Vehicle capacity.
    coordinates : NDArray or None
        (n+1, 2) coordinates for plotting.
    instance_name : str
        Name for logging.
    """

    def __init__(
        self,
        distance_matrix: NDArray,
        demands: NDArray,
        capacity: float,
        coordinates: NDArray | None = None,
        instance_name: str = "CVRP",
    ) -> None:
        self.distance_matrix = distance_matrix
        self.demands = np.asarray(demands, dtype=float)
        self.capacity = float(capacity)
        self.coordinates = coordinates
        self.instance_name = instance_name
        # Number of customers (excludes depot)
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
        n = len(all_coords)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(all_coords[i] - all_coords[j]))
                dist[i, j] = d
                dist[j, i] = d
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
        demands = [float(rng.integers(5, 20)) for _ in range(n_customers)]
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
