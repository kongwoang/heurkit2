"""
Presets — default algorithm configurations per problem type.
"""

from __future__ import annotations

from heurkit.algorithms.greedy import GreedyConstructor
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.core.runtime import SearchAlgorithm


def get_preset_algorithms(
    problem_type: str,
    time_limit: float = 5.0,
    seed: int | None = None,
) -> list[SearchAlgorithm]:
    """Return a list of default algorithms for a given problem type.

    Parameters
    ----------
    problem_type : str
        One of 'tsp', 'cvrp', 'binpacking'.
    time_limit : float
        Per-algorithm time limit in seconds.
    seed : int or None
        Random seed.

    Returns
    -------
    list[SearchAlgorithm]
    """
    half = time_limit / 2.0

    if problem_type == "tsp":
        return [
            HillClimbing(max_seconds=half, seed=seed, no_improvement_limit=1000),
            SimulatedAnnealing(max_seconds=time_limit, seed=seed),
        ]
    elif problem_type == "cvrp":
        return [
            HillClimbing(max_seconds=half, seed=seed, no_improvement_limit=500),
            TabuSearch(max_seconds=time_limit, seed=seed),
        ]
    elif problem_type == "binpacking":
        return [
            HillClimbing(max_seconds=half, seed=seed, no_improvement_limit=500),
            IteratedLocalSearch(max_seconds=time_limit, seed=seed),
        ]
    else:
        raise ValueError(
            f"Unknown problem type '{problem_type}'. "
            f"Supported: tsp, cvrp, binpacking"
        )
