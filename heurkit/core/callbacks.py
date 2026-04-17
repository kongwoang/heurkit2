"""
Simple callback system for search algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from heurkit.core.evaluator import Evaluation
    from heurkit.core.solution import Solution


class SearchCallback(Protocol):
    """Protocol for optional search callbacks."""

    def on_iteration(
        self,
        iteration: int,
        current: Solution,
        current_eval: Evaluation,
        best: Solution,
        best_eval: Evaluation,
    ) -> None: ...

    def on_new_best(
        self, iteration: int, solution: Solution, evaluation: Evaluation
    ) -> None: ...


class PrintCallback:
    """Simple callback that prints progress every *interval* iterations."""

    def __init__(self, interval: int = 100) -> None:
        self.interval = interval

    def on_iteration(
        self,
        iteration: int,
        current: Solution,
        current_eval: Evaluation,
        best: Solution,
        best_eval: Evaluation,
    ) -> None:
        if iteration % self.interval == 0:
            tag = "✓" if best_eval.is_feasible else "✗"
            print(f"  iter {iteration:>6d}  best={best_eval.objective:.4f} {tag}")

    def on_new_best(
        self, iteration: int, solution: Solution, evaluation: Evaluation
    ) -> None:
        pass
