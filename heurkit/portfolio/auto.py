"""
AutoSolver — runs preset algorithms and returns the best result.

Demonstrates a simple "scikit-learn-like" interface for heuristic
optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heurkit.core.result import SearchResult
from heurkit.portfolio.presets import get_preset_algorithms

if TYPE_CHECKING:
    from heurkit.core.problem import Problem


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
    """

    # Map Problem subclass names → problem_type keys
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
    ) -> None:
        self.problem_type = problem_type
        self.time_limit = time_limit
        self.seed = seed

    def _infer_type(self, problem: Problem) -> str:
        cls_name = type(problem).__name__
        pt = self._CLASS_MAP.get(cls_name)
        if pt is None:
            raise ValueError(
                f"Cannot infer problem type from {cls_name}. "
                f"Pass problem_type explicitly."
            )
        return pt

    def solve(self, problem: Problem) -> SearchResult:
        """Run preset algorithms and return the best result."""
        pt = self.problem_type or self._infer_type(problem)
        algorithms = get_preset_algorithms(
            pt, time_limit=self.time_limit, seed=self.seed
        )

        best_result: SearchResult | None = None

        for algo in algorithms:
            result = algo.solve(problem)
            if best_result is None or (
                result.is_feasible
                and result.best_objective < best_result.best_objective
            ):
                best_result = result

        assert best_result is not None
        best_result.metadata["auto_solver"] = True
        best_result.metadata["problem_type"] = pt
        return best_result
