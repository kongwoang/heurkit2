"""Runtime adapters that plug callback-defined custom problems into HeurKit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Iterator

from heurkit.core.evaluator import Evaluation, Evaluator
from heurkit.core.move import Move
from heurkit.core.random_state import make_rng
from heurkit.core.solution import Solution
from heurkit.custom.score import Score
from heurkit.custom.validation import (
    CallbackExecutionError,
    ensure_bool,
    ensure_iterable,
    ensure_not_none,
)

if TYPE_CHECKING:
    from numpy.random import Generator

    from heurkit.custom.problem import CustomProblem


class CustomSolution(Solution):
    """Opaque solution wrapper around a user-defined state object."""

    def __init__(
        self,
        state: Any,
        *,
        problem_data: Any,
        state_copy_callback,
        pretty_print_callback,
    ) -> None:
        self.state = state
        self._problem_data = problem_data
        self._state_copy_callback = state_copy_callback
        self._pretty_print_callback = pretty_print_callback

    def copy(self) -> CustomSolution:
        if self._state_copy_callback is not None:
            copied = self._run_callback(
                "state_copy",
                self._state_copy_callback,
                self.state,
            )
        else:
            copied = deepcopy(self.state)
        ensure_not_none("state_copy", copied, "state copy")
        return CustomSolution(
            copied,
            problem_data=self._problem_data,
            state_copy_callback=self._state_copy_callback,
            pretty_print_callback=self._pretty_print_callback,
        )

    def to_dict(self) -> dict[str, Any]:
        view: dict[str, Any] = {
            "state_type": type(self.state).__name__,
            "state": _serialise_state(self.state),
        }
        if self._pretty_print_callback is not None:
            try:
                rendered = self._pretty_print_callback(self._problem_data, self.state)
                if isinstance(rendered, str):
                    view["pretty"] = rendered
            except Exception:
                # pretty_print is optional and should not break serialisation.
                pass
        return view

    @staticmethod
    def _run_callback(name: str, callback, *args):
        try:
            return callback(*args)
        except Exception as exc:
            raise CallbackExecutionError(
                f"'{name}' callback raised an exception: {exc}"
            ) from exc


class CallbackMove(Move):
    """Move wrapper that delegates state transitions to callback functions."""

    def __init__(self, problem: CustomProblem, payload: Any, rng: Generator) -> None:
        self.problem = problem
        self.payload = payload
        self.rng = rng

    def apply(self, solution: CustomSolution) -> CustomSolution:  # type: ignore[override]
        if not isinstance(solution, CustomSolution):
            raise CallbackExecutionError(
                "Custom moves can only be applied to CustomSolution instances."
            )

        new_state = self._run_callback(
            "apply_move",
            self.problem.apply_move_callback,
            self.problem.problem_data,
            solution.state,
            self.payload,
        )
        ensure_not_none("apply_move", new_state, "new state")

        if self.problem.repair_callback is not None:
            repaired = self._run_callback(
                "repair",
                self.problem.repair_callback,
                self.problem.problem_data,
                new_state,
                self.rng,
            )
            new_state = ensure_not_none("repair", repaired, "repaired state")

        solution.state = new_state
        return solution

    def label(self) -> str:
        if hasattr(self.payload, "label") and callable(self.payload.label):
            try:
                return str(self.payload.label())
            except Exception:
                pass
        return repr(self.payload)

    @staticmethod
    def _run_callback(name: str, callback, *args):
        try:
            return callback(*args)
        except Exception as exc:
            raise CallbackExecutionError(
                f"'{name}' callback raised an exception: {exc}"
            ) from exc


class CallbackConstructor:
    """Constructor adapter for custom callback-defined problems."""

    def __init__(self, problem: CustomProblem, seed: int | None = None) -> None:
        self.problem = problem
        self.rng = make_rng(seed)

    def construct(self, problem: CustomProblem | None = None) -> CustomSolution:
        p = problem or self.problem
        state = self._run_callback(
            "constructor",
            p.constructor_callback,
            p.problem_data,
            self.rng,
        )
        state = ensure_not_none("constructor", state, "initial state")

        if p.repair_callback is not None:
            repaired = self._run_callback(
                "repair",
                p.repair_callback,
                p.problem_data,
                state,
                self.rng,
            )
            state = ensure_not_none("repair", repaired, "repaired state")

        return CustomSolution(
            state,
            problem_data=p.problem_data,
            state_copy_callback=p.state_copy_callback,
            pretty_print_callback=p.pretty_print_callback,
        )

    @staticmethod
    def _run_callback(name: str, callback, *args):
        try:
            return callback(*args)
        except Exception as exc:
            raise CallbackExecutionError(
                f"'{name}' callback raised an exception: {exc}"
            ) from exc


class CallbackEvaluator(Evaluator):
    """Evaluator adapter with support for float and Score objectives."""

    def __init__(self, problem: CustomProblem) -> None:
        self.problem = problem

    def evaluate(self, solution: Solution) -> Evaluation:
        if not isinstance(solution, CustomSolution):
            raise CallbackExecutionError(
                "Custom evaluator expects a CustomSolution instance."
            )

        raw_value = self._run_callback(
            "objective",
            self.problem.objective_callback,
            self.problem.problem_data,
            solution.state,
        )
        obj, hard_violations, soft_penalty = _parse_objective(raw_value)

        hard_penalty = self.problem.hard_violation_weight * max(0.0, hard_violations)
        penalty = hard_penalty + soft_penalty

        if self.problem.sense == "max":
            comparable = -obj + penalty
            display = obj - penalty
        else:
            comparable = obj + penalty
            display = obj + penalty

        feasible = hard_violations <= 0
        if self.problem.feasibility_callback is not None:
            raw_feasible = self._run_callback(
                "feasibility",
                self.problem.feasibility_callback,
                self.problem.problem_data,
                solution.state,
            )
            feasible = feasible and ensure_bool("feasibility", raw_feasible)

        details = {
            "raw_objective": obj,
            "hard_violations": hard_violations,
            "soft_penalty": soft_penalty,
            "penalty": penalty,
            "sense": self.problem.sense,
            "display_objective": display,
        }

        return Evaluation(objective=comparable, is_feasible=feasible, details=details)

    def objective_for_result(self, evaluation: Evaluation) -> float:
        """Convert internal comparable objective back to user-facing objective."""
        if evaluation.details and "display_objective" in evaluation.details:
            return float(evaluation.details["display_objective"])
        return float(evaluation.objective)

    @staticmethod
    def _run_callback(name: str, callback, *args):
        try:
            return callback(*args)
        except Exception as exc:
            raise CallbackExecutionError(
                f"'{name}' callback raised an exception: {exc}"
            ) from exc


class CallbackNeighborhood:
    """Neighbourhood adapter that wraps arbitrary move payloads."""

    def __init__(self, problem: CustomProblem, seed: int | None = None) -> None:
        self.problem = problem
        self.rng = make_rng(seed)

    def generate(self, solution: Solution) -> Iterator[Move]:
        if not isinstance(solution, CustomSolution):
            raise CallbackExecutionError(
                "Custom neighborhood expects a CustomSolution instance."
            )

        raw_moves = self._run_callback(
            "generate_moves",
            self.problem.generate_moves_callback,
            self.problem.problem_data,
            solution.state,
            self.rng,
        )
        move_iterable = ensure_iterable("generate_moves", raw_moves)

        for move_payload in move_iterable:
            ensure_not_none("generate_moves", move_payload, "move payload")
            yield CallbackMove(self.problem, move_payload, self.rng)

    @staticmethod
    def _run_callback(name: str, callback, *args):
        try:
            return callback(*args)
        except Exception as exc:
            raise CallbackExecutionError(
                f"'{name}' callback raised an exception: {exc}"
            ) from exc


def _parse_objective(value: Any) -> tuple[float, float, float]:
    if isinstance(value, Score):
        return (
            float(value.objective),
            float(value.hard_violations),
            float(value.soft_penalty),
        )

    if isinstance(value, (int, float)):
        return (float(value), 0.0, 0.0)

    raise CallbackExecutionError(
        "'objective' callback must return float/int or Score; "
        f"got {type(value).__name__}."
    )


def _serialise_state(state: Any) -> Any:
    if is_dataclass(state):
        try:
            return asdict(state)
        except Exception:
            return repr(state)
    if isinstance(state, (str, int, float, bool, type(None))):
        return state
    if isinstance(state, list):
        return [_serialise_state(v) for v in state]
    if isinstance(state, tuple):
        return tuple(_serialise_state(v) for v in state)
    if isinstance(state, dict):
        return {str(k): _serialise_state(v) for k, v in state.items()}
    if hasattr(state, "to_dict") and callable(state.to_dict):
        try:
            return state.to_dict()
        except Exception:
            return repr(state)
    return repr(state)
