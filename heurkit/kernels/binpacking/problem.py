"""
Bin Packing problem definition — 1D bin packing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from heurkit.core.problem import Problem
from heurkit.kernels.binpacking.evaluator import BinPackingEvaluator
from heurkit.kernels.binpacking.constructors import FirstFitDecreasingConstructor
from heurkit.kernels.binpacking.neighbors import BinPackingNeighborhood


class BinPackingProblem(Problem):
    """1-D Bin Packing Problem.

    Parameters
    ----------
    item_sizes : NDArray
        Size of each item.
    bin_capacity : float
        Capacity of every bin.
    instance_name : str
        Name for logging.
    """

    def __init__(
        self,
        item_sizes: NDArray,
        bin_capacity: float,
        instance_name: str = "BinPacking",
    ) -> None:
        self.item_sizes = np.asarray(item_sizes, dtype=float)
        self.bin_capacity = float(bin_capacity)
        self.instance_name = instance_name
        self.n_items: int = len(self.item_sizes)

    # ---- factory helpers ---------------------------------------------------

    @classmethod
    def from_sizes(
        cls,
        sizes: list[float],
        capacity: float,
        instance_name: str = "BinPacking",
    ) -> BinPackingProblem:
        return cls(np.array(sizes, dtype=float), capacity, instance_name)

    @classmethod
    def generate_random(
        cls,
        n_items: int = 30,
        capacity: float = 100.0,
        seed: int | None = None,
    ) -> BinPackingProblem:
        """Generate a random 1D bin packing instance."""
        rng = np.random.default_rng(seed)
        sizes = rng.integers(10, int(capacity * 0.6), size=n_items).astype(float)
        return cls(sizes, capacity, instance_name=f"BP-rand-{n_items}")

    # ---- Problem interface -------------------------------------------------

    def name(self) -> str:
        return self.instance_name

    def size(self) -> int:
        return self.n_items

    # ---- kernel defaults ---------------------------------------------------

    def default_evaluator(self) -> BinPackingEvaluator:
        return BinPackingEvaluator(self)

    def default_constructor(self) -> FirstFitDecreasingConstructor:
        return FirstFitDecreasingConstructor(self)

    def default_neighborhood(self) -> BinPackingNeighborhood:
        return BinPackingNeighborhood(self)
