"""
Iterated Local Search (ILS).

Alternates between local search (hill climbing) and perturbation
to escape local optima.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from heurkit.core.random_state import make_rng
from heurkit.core.result import SearchResult
from heurkit.core.runtime import SearchAlgorithm
from heurkit.core.stopping import StoppingCriteria

if TYPE_CHECKING:
    from heurkit.core.evaluator import Evaluator
    from heurkit.core.problem import Problem
    from heurkit.core.runtime import Constructor, NeighborhoodGenerator


class IteratedLocalSearch(SearchAlgorithm):
    """Iterated Local Search.

    Parameters
    ----------
    perturbation_strength : int
        Number of random moves applied as perturbation.
    local_search_iters : int
        Hill-climbing iterations per local search phase.
    max_iterations : int
        Number of ILS outer iterations.
    max_seconds : float
        Time limit.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        perturbation_strength: int = 5,
        local_search_iters: int = 200,
        max_iterations: int = 500,
        max_seconds: float = 10.0,
        no_improvement_limit: int = 100,
        seed: int | None = None,
        time_limit: float | None = None,
    ) -> None:
        self.perturbation_strength = perturbation_strength
        self.local_search_iters = local_search_iters
        self.stopping = StoppingCriteria(
            max_iterations=max_iterations,
            max_seconds=time_limit or max_seconds,
            no_improvement_iterations=no_improvement_limit,
        )
        self.seed = seed
        self.rng = make_rng(seed)

    def _local_search(
        self, solution, evaluator, neighborhood, max_iters: int
    ):
        """Run a quick first-improvement hill climb."""
        current = solution
        current_eval = evaluator.evaluate(current)

        for _ in range(max_iters):
            improved_inner = False
            for move in neighborhood.generate(current):
                candidate = current.copy()
                move.apply(candidate)
                cand_eval = evaluator.evaluate(candidate)
                if evaluator.is_better(cand_eval, current_eval):
                    current = candidate
                    current_eval = cand_eval
                    improved_inner = True
                    break
            if not improved_inner:
                break

        return current, current_eval

    def _perturb(self, solution, neighborhood):
        """Apply several random moves to escape the current basin."""
        perturbed = solution.copy()
        for _ in range(self.perturbation_strength):
            moves = list(neighborhood.generate(perturbed))
            if not moves:
                break
            idx = int(self.rng.integers(0, len(moves)))
            try:
                moves[idx].apply(perturbed)
            except (IndexError, ValueError):
                # Move may be invalid after structural changes; skip it
                continue
        return perturbed

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

        # Initial construction + local search
        initial = constructor.construct(problem)
        current, current_eval = self._local_search(
            initial, evaluator, neighborhood, self.local_search_iters
        )
        best = current.copy()
        best_eval = current_eval
        history: list[float] = [best_eval.objective]

        self.stopping.start()

        while not self.stopping.should_stop():
            # Perturb
            perturbed = self._perturb(current, neighborhood)
            # Local search
            candidate, cand_eval = self._local_search(
                perturbed, evaluator, neighborhood, self.local_search_iters
            )

            improved = False
            # Accept if better (simple acceptance criterion)
            if evaluator.is_better(cand_eval, current_eval):
                current = candidate
                current_eval = cand_eval

            if evaluator.is_better(cand_eval, best_eval):
                best = candidate.copy()
                best_eval = cand_eval
                improved = True

            self.stopping.step(improved)
            history.append(best_eval.objective)

        return SearchResult(
            algorithm_name="IteratedLocalSearch",
            problem_name=problem.name(),
            best_solution=best,
            best_objective=best_eval.objective,
            is_feasible=best_eval.is_feasible,
            iterations=self.stopping.iteration,
            runtime_seconds=self.stopping.elapsed,
            history=history,
        )
