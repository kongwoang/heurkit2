"""Score model for low-code custom problems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    """Structured objective output for custom problems.

    Parameters
    ----------
    objective : float
        Base objective value from the user model.
    hard_violations : int | float
        Count or magnitude of hard-constraint violations.
    soft_penalty : float
        Additional soft-constraint penalty.

    Notes
    -----
    ``total`` is a simple additive combination. The final comparable score used
    by the runtime is handled by :class:`heurkit.custom.adapters.CallbackEvaluator`.
    """

    objective: float
    hard_violations: int | float = 0
    soft_penalty: float = 0.0

    @property
    def total(self) -> float:
        """Return additive objective + penalties (without hard-weight scaling)."""
        return float(self.objective) + float(self.hard_violations) + float(self.soft_penalty)
