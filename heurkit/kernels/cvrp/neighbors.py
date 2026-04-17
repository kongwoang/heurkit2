"""
CVRP neighbourhood generator.
"""

from __future__ import annotations

from typing import Iterator, TYPE_CHECKING

from heurkit.core.move import Move
from heurkit.core.random_state import make_rng
from heurkit.kernels.cvrp.moves import (
    RelocateMove,
    SwapCustomersMove,
    IntraRouteTwoOptMove,
)

if TYPE_CHECKING:
    from heurkit.kernels.cvrp.problem import CVRPProblem
    from heurkit.kernels.cvrp.solution import CVRPSolution


class CVRPNeighborhood:
    """Randomly samples relocate, swap, and intra-route 2-opt moves."""

    def __init__(
        self,
        problem: CVRPProblem,
        seed: int | None = None,
        moves_per_call: int = 30,
    ) -> None:
        self.problem = problem
        self.rng = make_rng(seed)
        self.moves_per_call = moves_per_call

    def generate(self, solution: CVRPSolution) -> Iterator[Move]:  # type: ignore[override]
        routes = solution.routes
        n_routes = len(routes)
        if n_routes == 0:
            return

        for _ in range(self.moves_per_call):
            move_type = int(self.rng.integers(0, 3))

            if move_type == 0 and n_routes >= 1:
                # Relocate
                sr = int(self.rng.integers(0, n_routes))
                if not routes[sr]:
                    continue
                sp = int(self.rng.integers(0, len(routes[sr])))
                dr = int(self.rng.integers(0, n_routes))
                dp = int(self.rng.integers(0, max(1, len(routes[dr]) + 1)))
                yield RelocateMove(sr, sp, dr, dp)

            elif move_type == 1 and n_routes >= 1:
                # Swap
                ra = int(self.rng.integers(0, n_routes))
                rb = int(self.rng.integers(0, n_routes))
                if not routes[ra] or not routes[rb]:
                    continue
                pa = int(self.rng.integers(0, len(routes[ra])))
                pb = int(self.rng.integers(0, len(routes[rb])))
                if ra == rb and pa == pb:
                    continue
                yield SwapCustomersMove(ra, pa, rb, pb)

            elif move_type == 2:
                # Intra-route 2-opt
                ri = int(self.rng.integers(0, n_routes))
                if len(routes[ri]) < 3:
                    continue
                i = int(self.rng.integers(0, len(routes[ri])))
                j = int(self.rng.integers(0, len(routes[ri])))
                if i != j:
                    yield IntraRouteTwoOptMove(ri, min(i, j), max(i, j))
