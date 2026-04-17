"""
Bin Packing evaluator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heurkit.core.evaluator import Evaluation, Evaluator

if TYPE_CHECKING:
    from heurkit.core.solution import Solution
    from heurkit.kernels.binpacking.problem import BinPackingProblem


class BinPackingEvaluator(Evaluator):
    """Evaluate a bin-packing solution.

    Primary objective : minimise number of non-empty bins.
    Secondary penalty : capacity violations.
    """

    PENALTY_FACTOR = 1000.0

    def __init__(self, problem: BinPackingProblem) -> None:
        self.problem = problem

    def evaluate(self, solution: Solution) -> Evaluation:
        from heurkit.kernels.binpacking.solution import BinPackingSolution

        assert isinstance(solution, BinPackingSolution)
        p = self.problem
        sizes = p.item_sizes
        capacity = p.bin_capacity

        n_bins = 0
        total_overflow = 0.0
        assigned_items: set[int] = set()

        for bin_items in solution.bins:
            if not bin_items:
                continue
            n_bins += 1
            load = sum(sizes[i] for i in bin_items)
            if load > capacity:
                total_overflow += load - capacity
            for item in bin_items:
                assigned_items.add(item)

        # All items must be assigned exactly once
        all_assigned = assigned_items == set(range(p.n_items))
        is_feasible = all_assigned and total_overflow == 0.0

        # Objective: number of bins + penalty for violations
        objective = float(n_bins) + self.PENALTY_FACTOR * total_overflow

        return Evaluation(
            objective=objective,
            is_feasible=is_feasible,
            details={"n_bins": n_bins, "overflow": total_overflow},
        )
