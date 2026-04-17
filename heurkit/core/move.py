"""
Abstract move interface.

A Move is a small, reversible modification to a Solution.
Domain kernels define concrete moves (swap, 2-opt, relocate, …).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heurkit.core.solution import Solution


class Move(ABC):
    """Base class for neighbourhood moves."""

    @abstractmethod
    def apply(self, solution: Solution) -> Solution:
        """Apply this move to *solution* **in place** and return it."""

    @abstractmethod
    def label(self) -> str:
        """Short human-readable description, e.g. 'swap(3,7)'."""

    def __repr__(self) -> str:
        return self.label()
