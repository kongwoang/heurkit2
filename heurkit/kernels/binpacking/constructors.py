"""
Bin Packing constructors — First Fit, First Fit Decreasing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from heurkit.kernels.binpacking.solution import BinPackingSolution

if TYPE_CHECKING:
    from heurkit.kernels.binpacking.problem import BinPackingProblem


class FirstFitConstructor:
    """First Fit heuristic: items in original order."""

    def __init__(self, problem: BinPackingProblem) -> None:
        self.problem = problem

    def construct(self, problem: BinPackingProblem | None = None) -> BinPackingSolution:
        p = problem or self.problem
        sizes = p.item_sizes
        capacity = p.bin_capacity

        bins: list[list[int]] = []
        bin_loads: list[float] = []

        for item_idx in range(p.n_items):
            placed = False
            for b in range(len(bins)):
                if bin_loads[b] + sizes[item_idx] <= capacity:
                    bins[b].append(item_idx)
                    bin_loads[b] += sizes[item_idx]
                    placed = True
                    break
            if not placed:
                bins.append([item_idx])
                bin_loads.append(sizes[item_idx])

        return BinPackingSolution(bins)


class FirstFitDecreasingConstructor:
    """First Fit Decreasing: sort items largest-first, then first fit."""

    def __init__(self, problem: BinPackingProblem) -> None:
        self.problem = problem

    def construct(self, problem: BinPackingProblem | None = None) -> BinPackingSolution:
        p = problem or self.problem
        sizes = p.item_sizes
        capacity = p.bin_capacity

        # Sort indices by decreasing size
        order = list(np.argsort(-sizes))

        bins: list[list[int]] = []
        bin_loads: list[float] = []

        for item_idx in order:
            placed = False
            for b in range(len(bins)):
                if bin_loads[b] + sizes[item_idx] <= capacity:
                    bins[b].append(item_idx)
                    bin_loads[b] += sizes[item_idx]
                    placed = True
                    break
            if not placed:
                bins.append([item_idx])
                bin_loads.append(sizes[item_idx])

        return BinPackingSolution(bins)
