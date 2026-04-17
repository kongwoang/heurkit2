"""
Greedy constructor wrapper — thin algorithm adapter.

Runs the constructor once and returns the result as a SearchResult.
Useful for baselines and for the portfolio layer.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Sequence

from heurkit.core.result import SearchResult
from heurkit.core.runtime import SearchAlgorithm

if TYPE_CHECKING:
    from heurkit.core.callbacks import SearchCallback
    from heurkit.core.evaluator import Evaluator
    from heurkit.core.problem import Problem
    from heurkit.core.runtime import Constructor, NeighborhoodGenerator


class GreedyConstructor(SearchAlgorithm):
    """Wraps any Constructor as a one-shot algorithm.

    Parameters
    ----------
    seed : int or None
        Random seed (unused but kept for interface consistency).
    """

    def __init__(self, seed: int | None = None, **kwargs) -> None:
        self.seed = seed

    def solve(
        self,
        problem: Problem,
        *,
        constructor: Constructor | None = None,
        evaluator: Evaluator | None = None,
        neighborhood: NeighborhoodGenerator | None = None,
        callbacks: Sequence[SearchCallback] | None = None,
    ) -> SearchResult:
        constructor, evaluator, _ = self._resolve_components(
            problem, constructor, evaluator, neighborhood, seed=self.seed
        )

        t0 = time.perf_counter()
        solution = constructor.construct(problem)
        evaluation = evaluator.evaluate(solution)
        elapsed = time.perf_counter() - t0
        result_objective = self._objective_for_result(evaluator, evaluation)

        return SearchResult(
            algorithm_name="GreedyConstructor",
            problem_name=problem.name(),
            best_solution=solution,
            best_objective=result_objective,
            is_feasible=evaluation.is_feasible,
            iterations=1,
            runtime_seconds=elapsed,
            history=[result_objective],
            metadata={"comparable_best_objective": float(evaluation.objective)},
        )
