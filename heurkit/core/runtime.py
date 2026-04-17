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

import inspect
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator, Protocol, Sequence

if TYPE_CHECKING:
    from heurkit.core.callbacks import SearchCallback
    from heurkit.core.evaluator import Evaluation, Evaluator
    from heurkit.core.move import Move
    from heurkit.core.problem import Problem
    from heurkit.core.result import SearchResult
    from heurkit.core.solution import Solution


logger = logging.getLogger("heurkit")


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
        callbacks: Sequence[SearchCallback] | None = None,
    ) -> SearchResult:
        """Run the search and return a :class:`SearchResult`."""

    # ---- helper: fire callbacks ---------------------------------------------

    @staticmethod
    def _fire_iteration(
        callbacks: Sequence[SearchCallback],
        iteration: int,
        current: Solution,
        current_eval: Evaluation,
        best: Solution,
        best_eval: Evaluation,
    ) -> None:
        for cb in callbacks:
            cb.on_iteration(iteration, current, current_eval, best, best_eval)

    @staticmethod
    def _fire_new_best(
        callbacks: Sequence[SearchCallback],
        iteration: int,
        solution: Solution,
        evaluation: Evaluation,
    ) -> None:
        for cb in callbacks:
            cb.on_new_best(iteration, solution, evaluation)

    # ---- helper: resolve kernel defaults ------------------------------------

    @staticmethod
    def _resolve_components(
        problem: Problem,
        constructor: Constructor | None,
        evaluator: Evaluator | None,
        neighborhood: NeighborhoodGenerator | None,
        seed: int | None = None,
    ) -> tuple[Constructor, Evaluator, NeighborhoodGenerator]:
        """Resolve constructor / evaluator / neighbourhood.

        If not supplied explicitly, try ``problem.default_constructor()``,
        ``problem.default_evaluator()``, ``problem.default_neighborhood()``.
        """
        if constructor is None:
            constructor = SearchAlgorithm._call_problem_factory(
                problem, "default_constructor", seed=seed
            )
            if constructor is None:
                raise ValueError("No constructor provided and problem has no default.")
        if evaluator is None:
            evaluator = SearchAlgorithm._call_problem_factory(
                problem, "default_evaluator", seed=None
            )
            if evaluator is None:
                raise ValueError("No evaluator provided and problem has no default.")
        if neighborhood is None:
            neighborhood = SearchAlgorithm._call_problem_factory(
                problem, "default_neighborhood", seed=seed
            )
            if neighborhood is None:
                raise ValueError(
                    "No neighborhood provided and problem has no default."
                )
        return constructor, evaluator, neighborhood

    @staticmethod
    def _call_problem_factory(problem: Problem, factory_name: str, seed: int | None):
        factory = getattr(problem, factory_name, None)
        if not callable(factory):
            return None
        if seed is not None and SearchAlgorithm._accepts_seed(factory):
            return factory(seed=seed)
        return factory()

    @staticmethod
    def _accepts_seed(factory) -> bool:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return False

        if "seed" in signature.parameters:
            return True
        for param in signature.parameters.values():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return True
        return False

    @staticmethod
    def _objective_for_result(evaluator: Evaluator, evaluation: Evaluation) -> float:
        """Return objective value used in public SearchResult/history outputs."""
        if hasattr(evaluator, "objective_for_result"):
            objective_for_result = getattr(evaluator, "objective_for_result")
            if callable(objective_for_result):
                try:
                    return float(objective_for_result(evaluation))
                except Exception:
                    pass
        return float(evaluation.objective)
