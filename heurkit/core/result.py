"""
Structured search result with serialisation support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from heurkit.core.solution import Solution


@dataclass
class SearchResult:
    """Record of a completed search run.

    Attributes
    ----------
    algorithm_name : str
        Name of the algorithm that produced this result.
    problem_name : str
        Name / description of the problem instance.
    best_solution : Solution
        The best solution found.
    best_objective : float
        Objective value of the best solution.
    is_feasible : bool
        Whether the best solution is feasible.
    iterations : int
        Number of iterations executed.
    runtime_seconds : float
        Wall-clock time in seconds.
    history : list[float]
        Best-so-far objective at each recorded point.
    metadata : dict
        Any additional algorithm-specific information.
    """

    algorithm_name: str
    problem_name: str
    best_solution: Solution
    best_objective: float
    is_feasible: bool
    iterations: int
    runtime_seconds: float
    history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a multi-line human-readable summary."""
        feas = "✓" if self.is_feasible else "✗"
        lines = [
            f"SearchResult",
            f"  algorithm   : {self.algorithm_name}",
            f"  problem     : {self.problem_name}",
            f"  objective   : {self.best_objective:.4f}",
            f"  feasible    : {self.is_feasible} {feas}",
            f"  iterations  : {self.iterations}",
            f"  runtime (s) : {self.runtime_seconds:.3f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (solution excluded)."""
        return {
            "algorithm_name": self.algorithm_name,
            "problem_name": self.problem_name,
            "best_objective": float(self.best_objective),
            "is_feasible": bool(self.is_feasible),
            "iterations": self.iterations,
            "runtime_seconds": round(self.runtime_seconds, 6),
            "history_length": len(self.history),
            "metadata": {k: _safe_val(v) for k, v in self.metadata.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        return self.summary()


def _safe_val(v: Any) -> Any:
    """Coerce numpy / non-serialisable values to JSON-safe types."""
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)
