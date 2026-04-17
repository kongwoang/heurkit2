"""
CVRP move operators — relocate, swap, intra-route 2-opt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heurkit.core.move import Move

if TYPE_CHECKING:
    from heurkit.kernels.cvrp.solution import CVRPSolution


class RelocateMove(Move):
    """Move a customer from one route position to another."""

    def __init__(
        self, src_route: int, src_pos: int, dst_route: int, dst_pos: int
    ) -> None:
        self.src_route = src_route
        self.src_pos = src_pos
        self.dst_route = dst_route
        self.dst_pos = dst_pos

    def apply(self, solution: CVRPSolution) -> CVRPSolution:  # type: ignore[override]
        customer = solution.routes[self.src_route].pop(self.src_pos)
        # Adjust dst_pos if removing from the same route before the insertion point
        dst_pos = self.dst_pos
        if self.src_route == self.dst_route and self.src_pos < dst_pos:
            dst_pos -= 1
        solution.routes[self.dst_route].insert(dst_pos, customer)
        # Remove empty routes
        solution.routes = [r for r in solution.routes if r]
        return solution

    def label(self) -> str:
        return (
            f"relocate(r{self.src_route}[{self.src_pos}]"
            f"->r{self.dst_route}[{self.dst_pos}])"
        )


class SwapCustomersMove(Move):
    """Swap two customers between (possibly different) routes."""

    def __init__(
        self, route_a: int, pos_a: int, route_b: int, pos_b: int
    ) -> None:
        self.route_a = route_a
        self.pos_a = pos_a
        self.route_b = route_b
        self.pos_b = pos_b

    def apply(self, solution: CVRPSolution) -> CVRPSolution:  # type: ignore[override]
        ra, rb = solution.routes[self.route_a], solution.routes[self.route_b]
        ra[self.pos_a], rb[self.pos_b] = rb[self.pos_b], ra[self.pos_a]
        return solution

    def label(self) -> str:
        return (
            f"swap(r{self.route_a}[{self.pos_a}]"
            f"<->r{self.route_b}[{self.pos_b}])"
        )


class IntraRouteTwoOptMove(Move):
    """2-opt reversal within a single route."""

    def __init__(self, route_idx: int, i: int, j: int) -> None:
        self.route_idx = route_idx
        self.i = i
        self.j = j

    def apply(self, solution: CVRPSolution) -> CVRPSolution:  # type: ignore[override]
        lo, hi = min(self.i, self.j), max(self.i, self.j)
        route = solution.routes[self.route_idx]
        route[lo : hi + 1] = reversed(route[lo : hi + 1])
        return solution

    def label(self) -> str:
        return f"2opt(r{self.route_idx}[{self.i},{self.j}])"
