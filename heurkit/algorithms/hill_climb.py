"""
Hill Climbing — steepest-descent / first-improvement local search.

Completely domain-agnostic: works through Solution, Move, Evaluator,
and NeighborhoodGenerator interfaces.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from heurkit.core.result import SearchResult
from heurkit.core.runtime import SearchAlgorithm
from heurkit.core.stopping import StoppingCriteria

if TYPE_CHECKING:
    from heurkit.core.evaluator import Evaluator
    from heurkit.core.problem import Problem
    from heurkit.core.runtime import Constructor, NeighborhoodGenerator


class HillClimbing(SearchAlgorithm):
    """First-improvement hill climbing.

    Parameters
    ----------
    max_iterations : int
        Maximum number of outer iterations.
    max_seconds : float
        Time limit in seconds.
    no_improvement_limit : int
        Stop after this many consecutive non-improving iterations.
    seed : int or None
        Random seed (passed to constructor if needed).
    """

    def __init__(
        self,
        max_iterations: int = 5_000,
        max_seconds: float = 10.0,
        no_improvement_limit: int = 500,
        seed: int | None = None,
        time_limit: float | None = None,
    ) -> None:
        self.stopping = StoppingCriteria(
            max_iterations=max_iterations,
            max_seconds=time_limit or max_seconds,
            no_improvement_iterations=no_improvement_limit,
        )
        self.seed = seed

    def solve(
        self,
        problem: Problem,
        *,
        constructor: Constructor | None = None,
        evaluator: Evaluator | None = None,
        neighborhood: NeighborhoodGenerator | None = None,
    ) -> SearchResult:
        constructor, evaluator, neighborhood = self._resolve_components(
            problem, constructor, evaluator, neighborhood
        )

        # Build initial solution
        current = constructor.construct(problem)
        current_eval = evaluator.evaluate(current)
        best = current.copy()
        best_eval = current_eval
        history: list[float] = [best_eval.objective]

        self.stopping.start()

        while not self.stopping.should_stop():
            improved = False

            for move in neighborhood.generate(current):
                candidate = current.copy()
                move.apply(candidate)
                cand_eval = evaluator.evaluate(candidate)

                if evaluator.is_better(cand_eval, current_eval):
                    current = candidate
                    current_eval = cand_eval
                    if evaluator.is_better(current_eval, best_eval):
                        best = current.copy()
                        best_eval = current_eval
                        improved = True
                    break  # first improvement

            self.stopping.step(improved)
            history.append(best_eval.objective)

        return SearchResult(
            algorithm_name="HillClimbing",
            problem_name=problem.name(),
            best_solution=best,
            best_objective=best_eval.objective,
            is_feasible=best_eval.is_feasible,
            iterations=self.stopping.iteration,
            runtime_seconds=self.stopping.elapsed,
            history=history,
        )
