"""
TSP move operators — swap, 2-opt, insert.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heurkit.core.move import Move

if TYPE_CHECKING:
    from heurkit.kernels.tsp.solution import TSPSolution


class SwapMove(Move):
    """Swap two cities in the tour."""

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j

    def apply(self, solution: TSPSolution) -> TSPSolution:  # type: ignore[override]
        solution.tour[self.i], solution.tour[self.j] = (
            solution.tour[self.j],
            solution.tour[self.i],
        )
        return solution

    def label(self) -> str:
        return f"swap({self.i},{self.j})"


class TwoOptMove(Move):
    """Reverse a sub-segment of the tour (2-opt)."""

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j

    def apply(self, solution: TSPSolution) -> TSPSolution:  # type: ignore[override]
        # Reverse the segment between i and j (inclusive)
        lo, hi = min(self.i, self.j), max(self.i, self.j)
        solution.tour[lo : hi + 1] = reversed(solution.tour[lo : hi + 1])
        return solution

    def label(self) -> str:
        return f"2opt({self.i},{self.j})"


class InsertMove(Move):
    """Remove city at position *src* and insert it at position *dst*."""

    def __init__(self, src: int, dst: int) -> None:
        self.src = src
        self.dst = dst

    def apply(self, solution: TSPSolution) -> TSPSolution:  # type: ignore[override]
        city = solution.tour.pop(self.src)
        solution.tour.insert(self.dst, city)
        return solution

    def label(self) -> str:
        return f"insert({self.src}->{self.dst})"
