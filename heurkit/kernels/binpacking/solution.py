"""
Bin Packing solution representation.
"""

from __future__ import annotations

from typing import Any

from heurkit.core.solution import Solution


class BinPackingSolution(Solution):
    """A bin-packing assignment: list of bins, each containing item indices."""

    def __init__(self, bins: list[list[int]]) -> None:
        self.bins: list[list[int]] = bins

    def copy(self) -> BinPackingSolution:
        return BinPackingSolution([list(b) for b in self.bins])

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": [list(b) for b in self.bins],
            "n_bins": len(self.bins),
        }

    def cleanup(self) -> None:
        """Remove empty bins."""
        self.bins = [b for b in self.bins if b]
