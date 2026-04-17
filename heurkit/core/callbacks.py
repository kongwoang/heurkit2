"""
Callback system for search algorithms.

Callbacks are optional observers that can be attached to any algorithm.
They receive events during the search (iteration ticks, new-best updates)
without modifying the search logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from heurkit.core.evaluator import Evaluation
    from heurkit.core.solution import Solution


logger = logging.getLogger("heurkit")


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
    """Prints progress every *interval* iterations to stdout."""

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


class LoggingCallback:
    """Logs progress via Python's logging module (level=INFO)."""

    def __init__(self, interval: int = 500) -> None:
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
            logger.info(
                "iter=%d  best=%.4f  feasible=%s",
                iteration,
                best_eval.objective,
                best_eval.is_feasible,
            )

    def on_new_best(
        self, iteration: int, solution: Solution, evaluation: Evaluation
    ) -> None:
        logger.info(
            "NEW BEST at iter %d: obj=%.4f  feasible=%s",
            iteration,
            evaluation.objective,
            evaluation.is_feasible,
        )


class HistoryCallback:
    """Records every new-best event with iteration number."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def on_iteration(self, iteration, current, current_eval, best, best_eval) -> None:
        pass

    def on_new_best(self, iteration, solution, evaluation) -> None:
        self.events.append({
            "iteration": iteration,
            "objective": evaluation.objective,
            "is_feasible": evaluation.is_feasible,
        })
