"""
Bin Packing neighbourhood generator.
"""

from __future__ import annotations

from typing import Iterator, TYPE_CHECKING

from heurkit.core.move import Move
from heurkit.core.random_state import make_rng
from heurkit.kernels.binpacking.moves import MoveItemMove, SwapItemsMove

if TYPE_CHECKING:
    from heurkit.kernels.binpacking.problem import BinPackingProblem
    from heurkit.kernels.binpacking.solution import BinPackingSolution


class BinPackingNeighborhood:
    """Random mix of move-item and swap-item moves."""

    def __init__(
        self,
        problem: BinPackingProblem,
        seed: int | None = None,
        moves_per_call: int = 30,
    ) -> None:
        self.problem = problem
        self.rng = make_rng(seed)
        self.moves_per_call = moves_per_call

    def generate(self, solution: BinPackingSolution) -> Iterator[Move]:  # type: ignore[override]
        bins = solution.bins
        n_bins = len(bins)
        if n_bins < 2:
            return

        for _ in range(self.moves_per_call):
            move_type = int(self.rng.integers(0, 2))

            if move_type == 0:
                # Move item
                sb = int(self.rng.integers(0, n_bins))
                if not bins[sb]:
                    continue
                sp = int(self.rng.integers(0, len(bins[sb])))
                db = int(self.rng.integers(0, n_bins))
                if db == sb:
                    continue
                yield MoveItemMove(sb, sp, db)

            else:
                # Swap items
                ba = int(self.rng.integers(0, n_bins))
                bb = int(self.rng.integers(0, n_bins))
                if ba == bb or not bins[ba] or not bins[bb]:
                    continue
                pa = int(self.rng.integers(0, len(bins[ba])))
                pb = int(self.rng.integers(0, len(bins[bb])))
                yield SwapItemsMove(ba, pa, bb, pb)
