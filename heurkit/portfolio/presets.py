"""
Presets — default algorithm configurations per problem type.
"""

from __future__ import annotations

from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.algorithms.vns import VariableNeighborhoodSearch
from heurkit.core.runtime import SearchAlgorithm

# Registry of all known presets
PRESETS: dict[str, dict[str, type]] = {
    "tsp": {
        "HillClimbing": HillClimbing,
        "SimulatedAnnealing": SimulatedAnnealing,
        "TabuSearch": TabuSearch,
        "VNS": VariableNeighborhoodSearch,
    },
    "cvrp": {
        "HillClimbing": HillClimbing,
        "TabuSearch": TabuSearch,
        "SimulatedAnnealing": SimulatedAnnealing,
        "VNS": VariableNeighborhoodSearch,
    },
    "binpacking": {
        "HillClimbing": HillClimbing,
        "IteratedLocalSearch": IteratedLocalSearch,
        "SimulatedAnnealing": SimulatedAnnealing,
        "VNS": VariableNeighborhoodSearch,
    },
    "custom": {
        "HillClimbing": HillClimbing,
        "TabuSearch": TabuSearch,
        "SimulatedAnnealing": SimulatedAnnealing,
        "IteratedLocalSearch": IteratedLocalSearch,
        "VNS": VariableNeighborhoodSearch,
    },
}

# Default algorithm picks per problem type
DEFAULT_PICKS: dict[str, list[str]] = {
    "tsp": ["HillClimbing", "SimulatedAnnealing"],
    "cvrp": ["HillClimbing", "TabuSearch"],
    "binpacking": ["HillClimbing", "IteratedLocalSearch"],
    "custom": ["HillClimbing", "TabuSearch"],
}


def list_presets(problem_type: str | None = None) -> dict[str, list[str]]:
    """Return available algorithm names per problem type.

    Parameters
    ----------
    problem_type : str or None
        If given, return only that problem type's presets.
    """
    if problem_type:
        _validate_type(problem_type)
        return {problem_type: list(PRESETS[problem_type].keys())}
    return {k: list(v.keys()) for k, v in PRESETS.items()}


def get_preset_algorithms(
    problem_type: str,
    time_limit: float = 5.0,
    seed: int | None = None,
    picks: list[str] | None = None,
) -> list[SearchAlgorithm]:
    """Return configured algorithm instances for a problem type.

    Parameters
    ----------
    problem_type : str
        One of 'tsp', 'cvrp', 'binpacking', 'custom'.
    time_limit : float
        Per-algorithm time limit in seconds.
    seed : int or None
        Random seed.
    picks : list[str] or None
        Algorithm names to use.  If None, uses the default picks.

    Returns
    -------
    list[SearchAlgorithm]
    """
    _validate_type(problem_type)
    registry = PRESETS[problem_type]
    names = picks or DEFAULT_PICKS[problem_type]

    per_algo_time = time_limit / max(len(names), 1)
    algorithms: list[SearchAlgorithm] = []

    for name in names:
        cls = registry.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown algorithm '{name}' for {problem_type}. "
                f"Available: {list(registry.keys())}"
            )
        algorithms.append(cls(time_limit=per_algo_time, seed=seed))

    return algorithms


def _validate_type(problem_type: str) -> None:
    if problem_type not in PRESETS:
        raise ValueError(
            f"Unknown problem type '{problem_type}'. "
            f"Supported: {list(PRESETS.keys())}"
        )
