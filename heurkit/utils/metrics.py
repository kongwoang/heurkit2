"""
Metrics and summary helpers.
"""

from __future__ import annotations

from heurkit.core.result import SearchResult


def results_table(results: list[SearchResult]) -> str:
    """Format a list of SearchResults as a simple ASCII table."""
    header = f"{'Algorithm':<25s} {'Problem':<20s} {'Objective':>12s} {'Feasible':>8s} {'Iters':>8s} {'Time(s)':>8s}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        lines.append(
            f"{r.algorithm_name:<25s} {r.problem_name:<20s} "
            f"{r.best_objective:>12.2f} {'Yes' if r.is_feasible else 'No':>8s} "
            f"{r.iterations:>8d} {r.runtime_seconds:>8.3f}"
        )
    lines.append(sep)
    return "\n".join(lines)
