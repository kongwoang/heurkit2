"""
Simulated Annealing — generic implementation.

Accepts worse moves with a probability that decreases over time
(exponential cooling schedule).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Sequence

from heurkit.core.random_state import make_rng
from heurkit.core.result import SearchResult
from heurkit.core.runtime import SearchAlgorithm
from heurkit.core.stopping import StoppingCriteria

if TYPE_CHECKING:
    from heurkit.core.callbacks import SearchCallback
    from heurkit.core.evaluator import Evaluator
    from heurkit.core.problem import Problem
    from heurkit.core.runtime import Constructor, NeighborhoodGenerator

logger = logging.getLogger("heurkit.algorithms")


class SimulatedAnnealing(SearchAlgorithm):
    """Simulated Annealing with exponential cooling.

    Parameters
    ----------
    initial_temp : float
        Starting temperature.
    cooling_rate : float
        Multiplicative cooling factor per iteration (0 < rate < 1).
    min_temp : float
        Stop cooling below this temperature.
    max_iterations : int
        Iteration limit.
    max_seconds : float
        Time limit.
    no_improvement_limit : int
        Stagnation limit.
    seed : int or None
        Random seed.
    time_limit : float or None
        Alias for *max_seconds*.
    """

    def __init__(
        self,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.9995,
        min_temp: float = 0.01,
        max_iterations: int = 50_000,
        max_seconds: float = 10.0,
        no_improvement_limit: int = 5_000,
        seed: int | None = None,
        time_limit: float | None = None,
    ) -> None:
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.stopping = StoppingCriteria(
            max_iterations=max_iterations,
            max_seconds=time_limit if time_limit is not None else max_seconds,
            no_improvement_iterations=no_improvement_limit,
        )
        self.seed = seed
        self.rng = make_rng(seed)

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
            problem, constructor, evaluator, neighborhood, seed=self.seed
        )

        current = constructor.construct(problem)
        current_eval = evaluator.evaluate(current)
        best = current.copy()
        best_eval = current_eval
        history: list[float] = [self._objective_for_result(evaluator, best_eval)]

        temp = self.initial_temp
        logger.info("SA started on %s (T0=%.2f, obj=%.4f)", problem.name(), temp, best_eval.objective)
        self.stopping.start()

        while not self.stopping.should_stop():
            # Pick one random move from the neighbourhood
            moves = list(neighborhood.generate(current))
            if not moves:
                self.stopping.step(False)
                continue

            move = moves[int(self.rng.integers(0, len(moves)))]
            candidate = current.copy()
            move.apply(candidate)
            cand_eval = evaluator.evaluate(candidate)

            delta = cand_eval.objective - current_eval.objective
            accept = False
            if delta < 0:
                accept = True
            elif temp > self.min_temp:
                prob = math.exp(-delta / max(temp, 1e-12))
                if float(self.rng.random()) < prob:
                    accept = True

            improved = False
            if accept:
                current = candidate
                current_eval = cand_eval
                if evaluator.is_better(current_eval, best_eval):
                    best = current.copy()
                    best_eval = current_eval
                    improved = True
                    self._fire_new_best(cbs, self.stopping.iteration, best, best_eval)

            temp *= self.cooling_rate
            self.stopping.step(improved)
            self._fire_iteration(cbs, self.stopping.iteration, current, current_eval, best, best_eval)
            history.append(self._objective_for_result(evaluator, best_eval))

        logger.info("SA finished: obj=%.4f iters=%d T_final=%.6f", best_eval.objective, self.stopping.iteration, temp)

        return SearchResult(
            algorithm_name="SimulatedAnnealing",
            problem_name=problem.name(),
            best_solution=best,
            best_objective=self._objective_for_result(evaluator, best_eval),
            is_feasible=best_eval.is_feasible,
            iterations=self.stopping.iteration,
            runtime_seconds=self.stopping.elapsed,
            history=history,
            metadata={
                "final_temp": temp,
                "comparable_best_objective": float(best_eval.objective),
            },
        )
