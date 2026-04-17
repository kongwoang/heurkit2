"""
TSP solution representation — a tour permutation.
"""

from __future__ import annotations

from typing import Any

from heurkit.core.solution import Solution


class TSPSolution(Solution):
    """A TSP tour stored as a list of city indices (permutation)."""

    def __init__(self, tour: list[int]) -> None:
        self.tour: list[int] = tour

    def copy(self) -> TSPSolution:
        return TSPSolution(list(self.tour))

    def to_dict(self) -> dict[str, Any]:
        return {"tour": list(self.tour), "n_cities": len(self.tour)}
