"""Fluent builder API for low-code custom problems."""

from __future__ import annotations

from typing import Any, Callable, Literal

from heurkit.custom.problem import CustomProblem
from heurkit.custom.validation import (
    CustomProblemValidationError,
    validate_callback,
    validate_hard_weight,
    validate_problem_name,
    validate_sense,
)


_UNSET = object()


class ProblemBuilder:
    """Build a :class:`CustomProblem` with callback functions.

    Example
    -------
    .. code-block:: python

        problem = (
            ProblemBuilder("my_problem")
            .data({...})
            .constructor(make_initial)
            .objective(evaluate, sense="min")
            .moves(generate_moves, apply_move)
            .build()
        )
    """

    def __init__(self, name: str) -> None:
        validate_problem_name(name)
        self._name = name

        self._data: Any = _UNSET
        self._constructor_callback: Callable[..., Any] | None = None
        self._objective_callback: Callable[..., Any] | None = None
        self._generate_moves_callback: Callable[..., Any] | None = None
        self._apply_move_callback: Callable[..., Any] | None = None

        self._feasibility_callback: Callable[..., Any] | None = None
        self._repair_callback: Callable[..., Any] | None = None
        self._pretty_print_callback: Callable[..., Any] | None = None
        self._state_copy_callback: Callable[..., Any] | None = None

        self._sense: Literal["min", "max"] = "min"
        self._hard_violation_weight: float = 1_000_000.0
        self._seed: int | None = None

    def data(self, problem_data: Any) -> ProblemBuilder:
        """Set instance data passed to every callback."""
        self._data = problem_data
        return self

    def constructor(self, callback: Callable[..., Any]) -> ProblemBuilder:
        """Set initial-state constructor: ``(problem_data, rng) -> state``."""
        self._constructor_callback = callback
        return self

    def objective(
        self,
        callback: Callable[..., Any],
        *,
        sense: Literal["min", "max"] = "min",
        hard_violation_weight: float = 1_000_000.0,
    ) -> ProblemBuilder:
        """Set objective callback.

        Parameters
        ----------
        callback : callable
            ``(problem_data, state) -> float | Score``
        sense : {'min', 'max'}
            Optimisation direction.
        hard_violation_weight : float
            Multiplier for ``Score.hard_violations``.
        """
        validate_sense(sense)
        validate_hard_weight(hard_violation_weight)
        self._objective_callback = callback
        self._sense = sense
        self._hard_violation_weight = float(hard_violation_weight)
        return self

    def moves(
        self,
        generate_callback: Callable[..., Any],
        apply_callback: Callable[..., Any],
    ) -> ProblemBuilder:
        """Set move callbacks.

        ``generate_callback(problem_data, state, rng) -> iterable[move]``
        ``apply_callback(problem_data, state, move) -> new_state``
        """
        self._generate_moves_callback = generate_callback
        self._apply_move_callback = apply_callback
        return self

    def feasibility(self, callback: Callable[..., Any]) -> ProblemBuilder:
        """Set optional feasibility callback: ``(problem_data, state) -> bool``."""
        self._feasibility_callback = callback
        return self

    def repair(self, callback: Callable[..., Any]) -> ProblemBuilder:
        """Set optional repair callback: ``(problem_data, state, rng) -> state``."""
        self._repair_callback = callback
        return self

    def pretty_print(self, callback: Callable[..., Any]) -> ProblemBuilder:
        """Set optional state renderer: ``(problem_data, state) -> str``."""
        self._pretty_print_callback = callback
        return self

    def state_copy(self, callback: Callable[..., Any]) -> ProblemBuilder:
        """Set optional state copier: ``(state) -> state``."""
        self._state_copy_callback = callback
        return self

    def with_seed(self, seed: int | None) -> ProblemBuilder:
        """Set default seed for custom constructor/neighbourhood adapters."""
        self._seed = seed
        return self

    def build(self) -> CustomProblem:
        """Validate and return a :class:`CustomProblem`."""
        if self._data is _UNSET:
            raise CustomProblemValidationError(
                "Problem data is missing. Call .data(...) before .build()."
            )

        required = {
            "constructor": self._constructor_callback,
            "objective": self._objective_callback,
            "generate_moves": self._generate_moves_callback,
            "apply_move": self._apply_move_callback,
        }
        missing = [name for name, cb in required.items() if cb is None]
        if missing:
            missing_fmt = ", ".join(missing)
            raise CustomProblemValidationError(
                f"Missing required callbacks: {missing_fmt}."
            )

        assert self._constructor_callback is not None
        assert self._objective_callback is not None
        assert self._generate_moves_callback is not None
        assert self._apply_move_callback is not None

        validate_callback("constructor", self._constructor_callback, positional_arity=2)
        validate_callback("objective", self._objective_callback, positional_arity=2)
        validate_callback("generate_moves", self._generate_moves_callback, positional_arity=3)
        validate_callback("apply_move", self._apply_move_callback, positional_arity=3)

        if self._feasibility_callback is not None:
            validate_callback("feasibility", self._feasibility_callback, positional_arity=2)
        if self._repair_callback is not None:
            validate_callback("repair", self._repair_callback, positional_arity=3)
        if self._pretty_print_callback is not None:
            validate_callback("pretty_print", self._pretty_print_callback, positional_arity=2)
        if self._state_copy_callback is not None:
            validate_callback("state_copy", self._state_copy_callback, positional_arity=1)

        return CustomProblem(
            problem_name=self._name,
            problem_data=self._data,
            constructor_callback=self._constructor_callback,
            objective_callback=self._objective_callback,
            generate_moves_callback=self._generate_moves_callback,
            apply_move_callback=self._apply_move_callback,
            feasibility_callback=self._feasibility_callback,
            repair_callback=self._repair_callback,
            pretty_print_callback=self._pretty_print_callback,
            state_copy_callback=self._state_copy_callback,
            sense=self._sense,
            hard_violation_weight=self._hard_violation_weight,
            seed=self._seed,
        )
