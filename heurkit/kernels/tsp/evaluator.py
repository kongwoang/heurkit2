"""
TSP evaluator — total tour distance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heurkit.core.evaluator import Evaluation, Evaluator

if TYPE_CHECKING:
    from heurkit.core.solution import Solution
    from heurkit.kernels.tsp.problem import TSPProblem


class TSPEvaluator(Evaluator):
    """Evaluates a TSP tour by summing edge distances."""

    def __init__(self, problem: TSPProblem) -> None:
        self.problem = problem

    def evaluate(self, solution: Solution) -> Evaluation:
        from heurkit.kernels.tsp.solution import TSPSolution

        assert isinstance(solution, TSPSolution)
        tour = solution.tour
        dist = self.problem.distance_matrix
        n = len(tour)

        total = 0.0
        for i in range(n):
            total += dist[tour[i], tour[(i + 1) % n]]

        # feasibility: must be a valid permutation
        is_valid = set(tour) == set(range(self.problem.n_cities)) and len(tour) == self.problem.n_cities

        return Evaluation(objective=total, is_feasible=is_valid)
