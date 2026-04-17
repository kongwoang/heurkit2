"""
Bin Packing move operators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heurkit.core.move import Move

if TYPE_CHECKING:
    from heurkit.kernels.binpacking.solution import BinPackingSolution


class MoveItemMove(Move):
    """Move an item from one bin to another."""

    def __init__(self, src_bin: int, item_pos: int, dst_bin: int) -> None:
        self.src_bin = src_bin
        self.item_pos = item_pos
        self.dst_bin = dst_bin

    def apply(self, solution: BinPackingSolution) -> BinPackingSolution:  # type: ignore[override]
        item = solution.bins[self.src_bin].pop(self.item_pos)
        solution.bins[self.dst_bin].append(item)
        solution.cleanup()
        return solution

    def label(self) -> str:
        return f"move(bin{self.src_bin}[{self.item_pos}]->bin{self.dst_bin})"


class SwapItemsMove(Move):
    """Swap items between two bins."""

    def __init__(
        self, bin_a: int, pos_a: int, bin_b: int, pos_b: int
    ) -> None:
        self.bin_a = bin_a
        self.pos_a = pos_a
        self.bin_b = bin_b
        self.pos_b = pos_b

    def apply(self, solution: BinPackingSolution) -> BinPackingSolution:  # type: ignore[override]
        ba, bb = solution.bins[self.bin_a], solution.bins[self.bin_b]
        ba[self.pos_a], bb[self.pos_b] = bb[self.pos_b], ba[self.pos_a]
        return solution

    def label(self) -> str:
        return (
            f"swap(bin{self.bin_a}[{self.pos_a}]"
            f"<->bin{self.bin_b}[{self.pos_b}])"
        )
