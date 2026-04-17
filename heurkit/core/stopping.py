"""
Reusable stopping criteria.

Multiple criteria can be combined — the search stops when *any* of
them fires.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StoppingCriteria:
    """Composite stopping condition for search algorithms.

    Parameters
    ----------
    max_iterations : int or None
        Stop after this many iterations.
    max_seconds : float or None
        Stop after this many wall-clock seconds.
    no_improvement_iterations : int or None
        Stop if the best objective has not improved for this many iterations.
    """

    max_iterations: int | None = None
    max_seconds: float | None = None
    no_improvement_iterations: int | None = None

    # ---- internal state (managed by the runtime / algorithm) ----
    _start_time: float = field(default=0.0, init=False, repr=False)
    _iteration: int = field(default=0, init=False, repr=False)
    _iters_since_improvement: int = field(default=0, init=False, repr=False)

    def start(self) -> None:
        """Call once before the search loop begins."""
        self._start_time = time.perf_counter()
        self._iteration = 0
        self._iters_since_improvement = 0

    def step(self, improved: bool) -> None:
        """Call once per iteration."""
        self._iteration += 1
        if improved:
            self._iters_since_improvement = 0
        else:
            self._iters_since_improvement += 1

    def should_stop(self) -> bool:
        """Return True when any criterion has been met."""
        if self.max_iterations is not None and self._iteration >= self.max_iterations:
            return True
        if self.max_seconds is not None:
            elapsed = time.perf_counter() - self._start_time
            if elapsed >= self.max_seconds:
                return True
        if (
            self.no_improvement_iterations is not None
            and self._iters_since_improvement >= self.no_improvement_iterations
        ):
            return True
        return False

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start_time

    @property
    def iteration(self) -> int:
        return self._iteration
