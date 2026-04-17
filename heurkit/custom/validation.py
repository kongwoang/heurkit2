"""Validation helpers and user-facing exceptions for custom problems."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


_CALLBACK_HINTS: dict[str, str] = {
    "constructor": "(problem_data, rng)",
    "objective": "(problem_data, state)",
    "generate_moves": "(problem_data, state, rng)",
    "apply_move": "(problem_data, state, move)",
    "feasibility": "(problem_data, state)",
    "repair": "(problem_data, state, rng)",
    "pretty_print": "(problem_data, state)",
    "state_copy": "(state)",
}


class CustomProblemError(ValueError):
    """Base exception for the low-code custom problem layer."""


class CustomProblemValidationError(CustomProblemError):
    """Raised when a custom problem definition is invalid."""


class CallbackExecutionError(CustomProblemError):
    """Raised when a callback returns invalid data at runtime."""


def validate_problem_name(name: str) -> None:
    """Validate the custom problem name."""
    if not isinstance(name, str) or not name.strip():
        raise CustomProblemValidationError("Problem name must be a non-empty string.")


def validate_callback(
    name: str,
    callback: Callable[..., Any],
    *,
    positional_arity: int,
) -> None:
    """Validate that *callback* is callable and accepts the required arity."""
    if not callable(callback):
        raise CustomProblemValidationError(
            f"'{name}' callback must be callable, got {type(callback).__name__}."
        )

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        # Some callables don't expose signatures; accept them if callable.
        return

    probe_args = [object()] * positional_arity
    try:
        signature.bind(*probe_args)
    except TypeError as exc:
        hint = _CALLBACK_HINTS.get(name, f"({positional_arity} args)")
        raise CustomProblemValidationError(
            f"'{name}' callback must support signature {hint}; got {signature}."
        ) from exc


def ensure_bool(callback_name: str, value: Any) -> bool:
    """Ensure a callback returned a bool."""
    if not isinstance(value, bool):
        raise CallbackExecutionError(
            f"'{callback_name}' callback must return bool, got {type(value).__name__}."
        )
    return value


def ensure_not_none(callback_name: str, value: Any, what: str) -> Any:
    """Ensure callback output is not ``None``."""
    if value is None:
        raise CallbackExecutionError(
            f"'{callback_name}' callback returned None for {what}. "
            "Return a valid value."
        )
    return value


def ensure_iterable(callback_name: str, value: Any):
    """Ensure callback output is iterable."""
    try:
        iter(value)
    except TypeError as exc:
        raise CallbackExecutionError(
            f"'{callback_name}' callback must return an iterable of moves, "
            f"got {type(value).__name__}."
        ) from exc
    return value


def validate_sense(sense: str) -> None:
    """Validate objective sense."""
    if sense not in {"min", "max"}:
        raise CustomProblemValidationError(
            f"Unsupported sense '{sense}'. Use 'min' or 'max'."
        )


def validate_hard_weight(weight: float) -> None:
    """Validate hard-violation weight."""
    if weight <= 0:
        raise CustomProblemValidationError(
            "hard_violation_weight must be > 0."
        )
