"""
CVRP solution representation — list of routes.
"""

from __future__ import annotations

from typing import Any

from heurkit.core.solution import Solution


class CVRPSolution(Solution):
    """A CVRP solution stored as a list of routes.

    Each route is a list of customer indices (1-based; depot = 0 is implicit
    at start and end of every route).
    """

    def __init__(self, routes: list[list[int]]) -> None:
        self.routes: list[list[int]] = routes

    def copy(self) -> CVRPSolution:
        return CVRPSolution([list(r) for r in self.routes])

    def to_dict(self) -> dict[str, Any]:
        return {
            "routes": [list(r) for r in self.routes],
            "n_routes": len(self.routes),
        }

    def all_customers(self) -> set[int]:
        """Return the set of all customers served."""
        return {c for route in self.routes for c in route}
