"""
Abstract evaluator interface.

An Evaluator computes the objective value of a solution and checks
feasibility.  Each kernel provides its own Evaluator that understands
the domain-specific solution structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heurkit.core.problem import Problem
    from heurkit.core.solution import Solution


@dataclass(frozen=True)
class Evaluation:
    """Immutable result of evaluating a solution."""

    objective: float
    is_feasible: bool
    details: dict | None = None


class Evaluator(ABC):
    """Domain-specific evaluator bound to a Problem instance."""

    @abstractmethod
    def evaluate(self, solution: Solution) -> Evaluation:
        """Compute objective and feasibility for *solution*."""

    def is_better(self, a: Evaluation, b: Evaluation) -> bool:
        """Return True if *a* is strictly better than *b* (minimisation)."""
        if a.is_feasible and not b.is_feasible:
            return True
        if not a.is_feasible and b.is_feasible:
            return False
        return a.objective < b.objective
