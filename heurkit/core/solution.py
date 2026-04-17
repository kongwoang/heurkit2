"""
Abstract solution interface.

Each domain kernel defines its own concrete Solution holding
problem-specific data (tour, routes, bin assignments, …).
The runtime treats solutions as opaque objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Solution(ABC):
    """Base class for domain-specific solution representations."""

    @abstractmethod
    def copy(self) -> Solution:
        """Return a deep copy of this solution."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise the solution to a plain dictionary (for logging)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()})"
