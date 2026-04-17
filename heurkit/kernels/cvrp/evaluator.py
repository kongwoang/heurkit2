"""
CVRP evaluator — total distance with capacity-violation penalty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heurkit.core.evaluator import Evaluation, Evaluator

if TYPE_CHECKING:
    from heurkit.core.solution import Solution
    from heurkit.kernels.cvrp.problem import CVRPProblem


class CVRPEvaluator(Evaluator):
    """Evaluate a CVRP solution.

    Objective: total travel distance.
    Feasibility: all routes respect vehicle capacity, every customer
    is served exactly once.
    """

    PENALTY_FACTOR = 1000.0  # penalty per unit of capacity violation

    def __init__(self, problem: CVRPProblem) -> None:
        self.problem = problem

    def evaluate(self, solution: Solution) -> Evaluation:
        from heurkit.kernels.cvrp.solution import CVRPSolution

        assert isinstance(solution, CVRPSolution)
        p = self.problem
        dist = p.distance_matrix
        demands = p.demands
        depot = p.depot

        total_distance = 0.0
        total_overload = 0.0
        served: set[int] = set()

        for route in solution.routes:
            if not route:
                continue
            # distance: depot -> c1 -> c2 -> ... -> depot
            route_dist = dist[depot, route[0]]
            route_load = 0.0
            for idx in range(len(route)):
                c = route[idx]
                served.add(c)
                route_load += demands[c]
                if idx + 1 < len(route):
                    route_dist += dist[c, route[idx + 1]]
            route_dist += dist[route[-1], depot]
            total_distance += route_dist

            if route_load > p.capacity:
                total_overload += route_load - p.capacity

        # Check all customers served exactly once
        expected = set(range(1, p.n_customers + 1))
        all_served = served == expected
        is_feasible = all_served and total_overload == 0.0

        objective = total_distance + self.PENALTY_FACTOR * total_overload

        return Evaluation(
            objective=objective,
            is_feasible=is_feasible,
            details={
                "distance": total_distance,
                "overload": total_overload,
                "n_routes": len(solution.routes),
            },
        )
