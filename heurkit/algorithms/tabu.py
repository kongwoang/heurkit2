"""
Tabu Search — generic implementation.

Maintains a short-term memory of recent move labels to prevent cycling.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from heurkit.core.result import SearchResult
from heurkit.core.runtime import SearchAlgorithm
from heurkit.core.stopping import StoppingCriteria

if TYPE_CHECKING:
    from heurkit.core.evaluator import Evaluator
    from heurkit.core.problem import Problem
    from heurkit.core.runtime import Constructor, NeighborhoodGenerator


class TabuSearch(SearchAlgorithm):
    """Basic Tabu Search.

    Parameters
    ----------
    tabu_tenure : int
        Number of iterations a move stays tabu.
    max_iterations : int
        Iteration limit.
    max_seconds : float
        Time limit.
    no_improvement_limit : int
        Stagnation limit.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        tabu_tenure: int = 20,
        max_iterations: int = 10_000,
        max_seconds: float = 10.0,
        no_improvement_limit: int = 1_000,
        seed: int | None = None,
        time_limit: float | None = None,
    ) -> None:
        self.tabu_tenure = tabu_tenure
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

        current = constructor.construct(problem)
        current_eval = evaluator.evaluate(current)
        best = current.copy()
        best_eval = current_eval
        history: list[float] = [best_eval.objective]

        tabu_list: deque[str] = deque(maxlen=self.tabu_tenure)
        self.stopping.start()

        while not self.stopping.should_stop():
            best_move = None
            best_move_eval = None
            best_candidate = None

            for move in neighborhood.generate(current):
                label = move.label()
                candidate = current.copy()
                move.apply(candidate)
                cand_eval = evaluator.evaluate(candidate)

                # Accept if not tabu, or if it beats the global best (aspiration)
                is_tabu = label in tabu_list
                aspiration = evaluator.is_better(cand_eval, best_eval)

                if (not is_tabu or aspiration) and (
                    best_move_eval is None
                    or evaluator.is_better(cand_eval, best_move_eval)
                ):
                    best_move = move
                    best_move_eval = cand_eval
                    best_candidate = candidate

            improved = False
            if best_move is not None and best_candidate is not None and best_move_eval is not None:
                current = best_candidate
                current_eval = best_move_eval
                tabu_list.append(best_move.label())

                if evaluator.is_better(current_eval, best_eval):
                    best = current.copy()
                    best_eval = current_eval
                    improved = True

            self.stopping.step(improved)
            history.append(best_eval.objective)

        return SearchResult(
            algorithm_name="TabuSearch",
            problem_name=problem.name(),
            best_solution=best,
            best_objective=best_eval.objective,
            is_feasible=best_eval.is_feasible,
            iterations=self.stopping.iteration,
            runtime_seconds=self.stopping.elapsed,
            history=history,
        )
