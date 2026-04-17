"""
CVRP constructors — greedy sequential and random feasible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from heurkit.core.random_state import make_rng
from heurkit.kernels.cvrp.solution import CVRPSolution

if TYPE_CHECKING:
    from heurkit.kernels.cvrp.problem import CVRPProblem


class GreedySequentialConstructor:
    """Build routes one at a time, greedily adding the nearest feasible customer."""

    def __init__(self, problem: CVRPProblem) -> None:
        self.problem = problem

    def construct(self, problem: CVRPProblem | None = None) -> CVRPSolution:
        p = problem or self.problem
        dist = p.distance_matrix
        demands = p.demands
        capacity = p.capacity
        depot = p.depot

        unvisited = set(range(1, p.n_customers + 1))
        routes: list[list[int]] = []

        while unvisited:
            route: list[int] = []
            load = 0.0
            current = depot

            while True:
                # Find nearest feasible unvisited customer
                best_c, best_d = -1, float("inf")
                for c in unvisited:
                    if load + demands[c] <= capacity and dist[current, c] < best_d:
                        best_d = dist[current, c]
                        best_c = c
                if best_c == -1:
                    break
                route.append(best_c)
                unvisited.remove(best_c)
                load += demands[best_c]
                current = best_c

            if route:
                routes.append(route)

        return CVRPSolution(routes)


class RandomFeasibleConstructor:
    """Randomly assign customers to routes respecting capacity."""

    def __init__(self, problem: CVRPProblem, seed: int | None = None) -> None:
        self.problem = problem
        self.rng = make_rng(seed)

    def construct(self, problem: CVRPProblem | None = None) -> CVRPSolution:
        p = problem or self.problem
        demands = p.demands
        capacity = p.capacity

        customers = list(range(1, p.n_customers + 1))
        self.rng.shuffle(customers)

        routes: list[list[int]] = []
        current_route: list[int] = []
        current_load = 0.0

        for c in customers:
            if current_load + demands[c] > capacity:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_load = demands[c]
            else:
                current_route.append(c)
                current_load += demands[c]

        if current_route:
            routes.append(current_route)

        return CVRPSolution(routes)
