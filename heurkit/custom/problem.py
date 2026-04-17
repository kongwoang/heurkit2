"""CustomProblem model for low-code callback-defined optimization tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from numpy.random import Generator
from heurkit.core.problem import Problem
from heurkit.custom.adapters import CallbackConstructor, CallbackEvaluator, CallbackNeighborhood
from heurkit.custom.score import Score


ConstructorCallback = Callable[[Any, Generator], Any]
ObjectiveCallback = Callable[[Any, Any], float | int | Score]
GenerateMovesCallback = Callable[[Any, Any, Generator], Iterable[Any]]
ApplyMoveCallback = Callable[[Any, Any, Any], Any]
FeasibilityCallback = Callable[[Any, Any], bool]
RepairCallback = Callable[[Any, Any, Generator], Any]
PrettyPrintCallback = Callable[[Any, Any], str]
StateCopyCallback = Callable[[Any], Any]


@dataclass
class CustomProblem(Problem):
    """A callback-defined optimization problem compatible with HeurKit runtime."""

    problem_name: str
    problem_data: Any
    constructor_callback: ConstructorCallback
    objective_callback: ObjectiveCallback
    generate_moves_callback: GenerateMovesCallback
    apply_move_callback: ApplyMoveCallback
    feasibility_callback: FeasibilityCallback | None = None
    repair_callback: RepairCallback | None = None
    pretty_print_callback: PrettyPrintCallback | None = None
    state_copy_callback: StateCopyCallback | None = None
    sense: Literal["min", "max"] = "min"
    hard_violation_weight: float = 1_000_000.0
    seed: int | None = None

    def name(self) -> str:
        return self.problem_name

    def size(self) -> int:
        data = self.problem_data
        if hasattr(data, "__len__"):
            try:
                return int(len(data))
            except Exception:
                pass
        return 1

    def metadata(self) -> dict[str, Any]:
        meta = super().metadata()
        meta.update(
            {
                "problem_type": "custom",
                "sense": self.sense,
                "has_feasibility": self.has_feasibility,
                "has_repair": self.has_repair,
            }
        )
        return meta

    @property
    def has_feasibility(self) -> bool:
        return self.feasibility_callback is not None

    @property
    def has_repair(self) -> bool:
        return self.repair_callback is not None

    def default_constructor(self, seed: int | None = None) -> CallbackConstructor:
        """Return the runtime constructor adapter."""
        return CallbackConstructor(self, seed=self._resolve_seed(seed))

    def default_evaluator(self) -> CallbackEvaluator:
        """Return the runtime evaluator adapter."""
        return CallbackEvaluator(self)

    def default_neighborhood(self, seed: int | None = None) -> CallbackNeighborhood:
        """Return the runtime neighbourhood adapter."""
        return CallbackNeighborhood(self, seed=self._resolve_seed(seed))

    def _resolve_seed(self, seed: int | None) -> int | None:
        if seed is None:
            return self.seed
        return seed
