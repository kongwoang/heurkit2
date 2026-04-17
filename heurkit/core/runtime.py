"""
Generic search runtime.

This module defines the abstract SearchAlgorithm base class, the
Constructor protocol, and the NeighborhoodGenerator protocol.

The runtime is **completely domain-agnostic** — it never references
cities, routes, bins, or any other domain concept.  It interacts with
domain kernels exclusively through the abstract interfaces defined in
heurkit.core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator, Protocol

if TYPE_CHECKING:
    from heurkit.core.evaluator import Evaluator
    from heurkit.core.move import Move
    from heurkit.core.problem import Problem
    from heurkit.core.result import SearchResult
    from heurkit.core.solution import Solution


# ---------------------------------------------------------------------------
# Protocols for constructors and neighbourhood generators
# ---------------------------------------------------------------------------

class Constructor(Protocol):
    """Builds an initial solution for a problem."""

    def construct(self, problem: Problem) -> Solution: ...


class NeighborhoodGenerator(Protocol):
    """Yields candidate moves for a given solution."""

    def generate(self, solution: Solution) -> Iterator[Move]: ...


# ---------------------------------------------------------------------------
# Abstract search algorithm
# ---------------------------------------------------------------------------

class SearchAlgorithm(ABC):
    """Base class for all search algorithms in HeurKit.

    Subclasses implement ``solve`` using the generic interfaces:
    Problem, Solution, Move, Evaluator, Constructor, NeighborhoodGenerator.

    The runtime never inspects domain-specific internals.
    """

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        *,
        constructor: Constructor | None = None,
        evaluator: Evaluator | None = None,
        neighborhood: NeighborhoodGenerator | None = None,
    ) -> SearchResult:
        """Run the search and return a :class:`SearchResult`."""

    # Convenience: allow algorithms to resolve defaults from a problem
    # if the problem carries kernel-provided defaults.
    @staticmethod
    def _resolve_components(
        problem: Problem,
        constructor: Constructor | None,
        evaluator: Evaluator | None,
        neighborhood: NeighborhoodGenerator | None,
    ) -> tuple[Constructor, Evaluator, NeighborhoodGenerator]:
        """Resolve constructor / evaluator / neighbourhood.

        If not supplied explicitly, try ``problem.default_constructor()``,
        ``problem.default_evaluator()``, ``problem.default_neighborhood()``.
        """
        if constructor is None:
            constructor = getattr(problem, "default_constructor", lambda: None)()
            if constructor is None:
                raise ValueError("No constructor provided and problem has no default.")
        if evaluator is None:
            evaluator = getattr(problem, "default_evaluator", lambda: None)()
            if evaluator is None:
                raise ValueError("No evaluator provided and problem has no default.")
        if neighborhood is None:
            neighborhood = getattr(problem, "default_neighborhood", lambda: None)()
            if neighborhood is None:
                raise ValueError(
                    "No neighborhood provided and problem has no default."
                )
        return constructor, evaluator, neighborhood
