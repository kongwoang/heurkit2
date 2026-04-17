"""
Abstract problem interface.

Every domain kernel must implement a concrete Problem subclass.
The runtime never inspects problem internals — it only passes the
problem object to evaluators, constructors, and move generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Problem(ABC):
    """Base class for all optimisation problem definitions.

    A Problem holds the *instance data* (distances, demands, capacities, …)
    and provides factory helpers to create instances from various sources.
    """

    @abstractmethod
    def name(self) -> str:
        """Short human-readable name, e.g. 'TSP-42' or 'CVRP-100'."""

    @abstractmethod
    def size(self) -> int:
        """Numeric indication of problem size (cities, items, …)."""

    def metadata(self) -> dict[str, Any]:
        """Optional metadata for logging / results."""
        return {"name": self.name(), "size": self.size()}
