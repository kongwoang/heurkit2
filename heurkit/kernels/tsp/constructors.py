"""
TSP constructors — random and nearest-neighbour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from heurkit.core.random_state import make_rng
from heurkit.kernels.tsp.solution import TSPSolution

if TYPE_CHECKING:
    from heurkit.kernels.tsp.problem import TSPProblem


class RandomConstructor:
    """Builds a random tour permutation."""

    def __init__(self, problem: TSPProblem, seed: int | None = None) -> None:
        self.problem = problem
        self.rng = make_rng(seed)

    def construct(self, problem: TSPProblem | None = None) -> TSPSolution:
        p = problem or self.problem
        tour = list(self.rng.permutation(p.n_cities))
        return TSPSolution(tour)


class NearestNeighborConstructor:
    """Greedy nearest-neighbour tour construction."""

    def __init__(self, problem: TSPProblem, start_city: int = 0) -> None:
        self.problem = problem
        self.start_city = start_city

    def construct(self, problem: TSPProblem | None = None) -> TSPSolution:
        p = problem or self.problem
        dist = p.distance_matrix
        n = p.n_cities
        visited = [False] * n
        tour: list[int] = [self.start_city]
        visited[self.start_city] = True

        for _ in range(n - 1):
            current = tour[-1]
            best_next = -1
            best_dist = float("inf")
            for j in range(n):
                if not visited[j] and dist[current, j] < best_dist:
                    best_dist = dist[current, j]
                    best_next = j
            tour.append(best_next)
            visited[best_next] = True

        return TSPSolution(tour)
