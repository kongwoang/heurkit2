"""
Hill Climbing — first-improvement local search.

Completely domain-agnostic: works through Solution, Move, Evaluator,
and NeighborhoodGenerator interfaces.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from heurkit.core.result import SearchResult
from heurkit.core.runtime import SearchAlgorithm
from heurkit.core.stopping import StoppingCriteria

if TYPE_CHECKING:
    from heurkit.core.callbacks import SearchCallback
    from heurkit.core.evaluator import Evaluator
    from heurkit.core.problem import Problem
    from heurkit.core.runtime import Constructor, NeighborhoodGenerator

logger = logging.getLogger("heurkit.algorithms")


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
    time_limit : float or None
        Alias for *max_seconds* (takes precedence if set).
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
            max_seconds=time_limit if time_limit is not None else max_seconds,
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
        callbacks: Sequence[SearchCallback] | None = None,
    ) -> SearchResult:
        cbs = callbacks or []
        constructor, evaluator, neighborhood = self._resolve_components(
            problem, constructor, evaluator, neighborhood
        )

        # Build initial solution
        current = constructor.construct(problem)
        current_eval = evaluator.evaluate(current)
        best = current.copy()
        best_eval = current_eval
        history: list[float] = [best_eval.objective]

        logger.info("HillClimbing started on %s (obj=%.4f)", problem.name(), best_eval.objective)
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
                        self._fire_new_best(cbs, self.stopping.iteration, best, best_eval)
                    break  # first improvement

            self.stopping.step(improved)
            self._fire_iteration(cbs, self.stopping.iteration, current, current_eval, best, best_eval)
            history.append(best_eval.objective)

        logger.info("HillClimbing finished: obj=%.4f iters=%d", best_eval.objective, self.stopping.iteration)

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
