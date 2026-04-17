"""
Variable Neighbourhood Search (VNS).

Systematically changes neighbourhood structures to escape local
optima.  When a neighbourhood fails to improve, VNS moves to the
next one.  When an improvement is found, it restarts from the
first (usually smallest) neighbourhood.

This is a *basic* VNS (BVNS) — the local search phase uses
first-improvement hill climbing within the current neighbourhood.
"""

from __future__ import annotations

import logging
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


class VariableNeighborhoodSearch(SearchAlgorithm):
    """Basic Variable Neighbourhood Search (BVNS).

    Parameters
    ----------
    neighborhoods : list[NeighborhoodGenerator] or None
        Ordered list of neighbourhood structures (small → large).
        If ``None``, the single default neighbourhood from the problem
        is used (degenerates to hill climbing with perturbation).
    local_search_iters : int
        Maximum hill-climb iterations per local search phase.
    max_iterations : int
        Outer iteration limit (one iteration = one shaking + LS cycle).
    max_seconds : float
        Wall-clock time limit.
    no_improvement_limit : int
        Stagnation limit.
    seed : int or None
        Random seed.
    time_limit : float or None
        Alias for *max_seconds*.
    """

    def __init__(
        self,
        neighborhoods: list[NeighborhoodGenerator] | None = None,
        local_search_iters: int = 200,
        max_iterations: int = 2_000,
        max_seconds: float = 10.0,
        no_improvement_limit: int = 300,
        seed: int | None = None,
        time_limit: float | None = None,
    ) -> None:
        self._neighborhoods = neighborhoods
        self.local_search_iters = local_search_iters
        self.stopping = StoppingCriteria(
            max_iterations=max_iterations,
            max_seconds=time_limit if time_limit is not None else max_seconds,
            no_improvement_iterations=no_improvement_limit,
        )
        self.seed = seed
        self.rng = make_rng(seed)

    # ---- internal helpers --------------------------------------------------

    def _shake(self, solution, neighborhood, strength: int = 3):
        """Random perturbation using *neighborhood*."""
        perturbed = solution.copy()
        for _ in range(strength):
            moves = list(neighborhood.generate(perturbed))
            if not moves:
                break
            idx = int(self.rng.integers(0, len(moves)))
            try:
                moves[idx].apply(perturbed)
            except (IndexError, ValueError):
                continue
        return perturbed

    def _local_search(self, solution, evaluator, neighborhood, max_iters: int):
        """First-improvement hill climb within a single neighbourhood."""
        current = solution
        current_eval = evaluator.evaluate(current)

        for _ in range(max_iters):
            improved = False
            for move in neighborhood.generate(current):
                candidate = current.copy()
                move.apply(candidate)
                cand_eval = evaluator.evaluate(candidate)
                if evaluator.is_better(cand_eval, current_eval):
                    current = candidate
                    current_eval = cand_eval
                    improved = True
                    break
            if not improved:
                break

        return current, current_eval

    # ---- main solve --------------------------------------------------------

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
        constructor, evaluator, default_nbr = self._resolve_components(
            problem, constructor, evaluator, neighborhood
        )

        # Build the list of neighbourhoods
        if self._neighborhoods:
            nbrs = list(self._neighborhoods)
        else:
            nbrs = [default_nbr]

        k_max = len(nbrs)

        # Initial solution
        current = constructor.construct(problem)
        current_eval = evaluator.evaluate(current)
        best = current.copy()
        best_eval = current_eval
        history: list[float] = [best_eval.objective]

        logger.info(
            "VNS started on %s (k_max=%d, obj=%.4f)",
            problem.name(), k_max, best_eval.objective,
        )
        self.stopping.start()

        while not self.stopping.should_stop():
            k = 0  # start from first neighbourhood
            improved_outer = False

            while k < k_max and not self.stopping.should_stop():
                # 1. Shaking — random perturbation in neighbourhood k
                #    Strength increases with neighbourhood index
                shaken = self._shake(current, nbrs[k], strength=k + 2)

                # 2. Local search — hill climb using neighbourhood k
                candidate, cand_eval = self._local_search(
                    shaken, evaluator, nbrs[k], self.local_search_iters
                )

                # 3. Move-or-not decision
                if evaluator.is_better(cand_eval, current_eval):
                    current = candidate
                    current_eval = cand_eval
                    k = 0  # restart from first neighbourhood

                    if evaluator.is_better(current_eval, best_eval):
                        best = current.copy()
                        best_eval = current_eval
                        improved_outer = True
                        self._fire_new_best(cbs, self.stopping.iteration, best, best_eval)
                else:
                    k += 1  # try next neighbourhood

            self.stopping.step(improved_outer)
            self._fire_iteration(cbs, self.stopping.iteration, current, current_eval, best, best_eval)
            history.append(best_eval.objective)

        logger.info("VNS finished: obj=%.4f iters=%d", best_eval.objective, self.stopping.iteration)

        return SearchResult(
            algorithm_name="VNS",
            problem_name=problem.name(),
            best_solution=best,
            best_objective=best_eval.objective,
            is_feasible=best_eval.is_feasible,
            iterations=self.stopping.iteration,
            runtime_seconds=self.stopping.elapsed,
            history=history,
            metadata={"k_max": k_max},
        )
