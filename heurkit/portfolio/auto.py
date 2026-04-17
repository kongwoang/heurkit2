"""
AutoSolver — runs preset algorithms and returns the best result.

Demonstrates a simple "scikit-learn-like" interface for heuristic
optimization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from heurkit.core.result import SearchResult
from heurkit.portfolio.presets import get_preset_algorithms, list_presets

if TYPE_CHECKING:
    from heurkit.core.problem import Problem

logger = logging.getLogger("heurkit.auto")


class AutoSolver:
    """Automatic solver that picks and runs preset algorithms.

    Usage
    -----
    >>> result = AutoSolver(time_limit=3.0, seed=42).solve(problem)

    Parameters
    ----------
    problem_type : str or None
        'tsp', 'cvrp', or 'binpacking'.  If None, inferred from
        the Problem class name.
    time_limit : float
        Total time budget in seconds (split across algorithms).
    seed : int or None
        Random seed for reproducibility.
    picks : list[str] or None
        Override the default algorithm picks.  Pass algorithm names
        like ``["HillClimbing", "TabuSearch"]``.
    return_all : bool
        If True, ``solve()`` stores all trial results in
        ``result.metadata["all_results"]``.
    """

    _CLASS_MAP: dict[str, str] = {
        "TSPProblem": "tsp",
        "CVRPProblem": "cvrp",
        "BinPackingProblem": "binpacking",
    }

    def __init__(
        self,
        problem_type: str | None = None,
        time_limit: float = 5.0,
        seed: int | None = None,
        picks: list[str] | None = None,
        return_all: bool = False,
    ) -> None:
        self.problem_type = problem_type
        self.time_limit = time_limit
        self.seed = seed
        self.picks = picks
        self.return_all = return_all

    def _infer_type(self, problem: Problem) -> str:
        cls_name = type(problem).__name__
        pt = self._CLASS_MAP.get(cls_name)
        if pt is None:
            raise ValueError(
                f"Cannot infer problem type from {cls_name}. "
                f"Pass problem_type explicitly."
            )
        return pt

    @staticmethod
    def available_presets(problem_type: str | None = None) -> dict[str, list[str]]:
        """List available algorithm presets per problem type."""
        return list_presets(problem_type)

    def solve(self, problem: Problem) -> SearchResult:
        """Run preset algorithms and return the best result."""
        pt = self.problem_type or self._infer_type(problem)
        algorithms = get_preset_algorithms(
            pt, time_limit=self.time_limit, seed=self.seed, picks=self.picks
        )

        all_results: list[SearchResult] = []
        best_result: SearchResult | None = None

        for algo in algorithms:
            logger.info("AutoSolver: running %s on %s", type(algo).__name__, problem.name())
            result = algo.solve(problem)
            all_results.append(result)

            if best_result is None or (
                result.is_feasible
                and result.best_objective < best_result.best_objective
            ):
                best_result = result

        assert best_result is not None
        best_result.metadata["auto_solver"] = True
        best_result.metadata["problem_type"] = pt
        best_result.metadata["n_trials"] = len(all_results)
        if self.return_all:
            best_result.metadata["all_results"] = [r.to_dict() for r in all_results]

        logger.info(
            "AutoSolver: best=%s obj=%.4f",
            best_result.algorithm_name,
            best_result.best_objective,
        )
        return best_result
