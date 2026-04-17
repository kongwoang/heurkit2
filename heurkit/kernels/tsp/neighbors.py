"""
TSP neighbourhood generator — yields moves for a given solution.
"""

from __future__ import annotations

from typing import Iterator, TYPE_CHECKING

from heurkit.core.move import Move
from heurkit.core.random_state import make_rng
from heurkit.kernels.tsp.moves import SwapMove, TwoOptMove, InsertMove

if TYPE_CHECKING:
    from heurkit.kernels.tsp.problem import TSPProblem
    from heurkit.kernels.tsp.solution import TSPSolution


class TSPNeighborhood:
    """Generates random moves from a mix of swap, 2-opt, and insert."""

    def __init__(
        self,
        problem: TSPProblem,
        seed: int | None = None,
        moves_per_call: int = 30,
    ) -> None:
        self.problem = problem
        self.rng = make_rng(seed)
        self.moves_per_call = moves_per_call

    def generate(self, solution: TSPSolution) -> Iterator[Move]:  # type: ignore[override]
        n = self.problem.n_cities
        for _ in range(self.moves_per_call):
            move_type = int(self.rng.integers(0, 3))
            i, j = int(self.rng.integers(0, n)), int(self.rng.integers(0, n))
            while i == j:
                j = int(self.rng.integers(0, n))
            if move_type == 0:
                yield SwapMove(i, j)
            elif move_type == 1:
                yield TwoOptMove(min(i, j), max(i, j))
            else:
                yield InsertMove(i, j)
