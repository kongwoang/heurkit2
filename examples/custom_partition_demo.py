#!/usr/bin/env python3
"""Custom partitioning demo (bin-like but not using built-in BinPacking kernel)."""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.custom import ProblemBuilder, Score
from heurkit.portfolio.auto import AutoSolver


@dataclass
class PartitionState:
    groups: list[int]


def make_initial(problem_data, rng):
    groups = [int(rng.integers(0, problem_data["k"])) for _ in problem_data["weights"]]
    return PartitionState(groups=groups)


def evaluate(problem_data, state):
    loads = [0.0] * problem_data["k"]
    for i, g in enumerate(state.groups):
        loads[g] += problem_data["weights"][i]

    mean_load = sum(loads) / len(loads)
    imbalance = sum(abs(load - mean_load) for load in loads)
    hard_violations = sum(1 for load in loads if load > problem_data["max_group_load"])

    return Score(
        objective=imbalance,
        hard_violations=hard_violations,
        soft_penalty=0.0,
    )


def generate_moves(problem_data, state, rng):
    n = len(state.groups)
    for _ in range(max(1, n * 2)):
        i = int(rng.integers(0, n))
        b = int(rng.integers(0, problem_data["k"]))
        if b != state.groups[i]:
            yield (i, b)


def apply_move(problem_data, state, move):
    i, b = move
    groups = state.groups.copy()
    groups[i] = b
    return PartitionState(groups=groups)


def main() -> None:
    problem = (
        ProblemBuilder("custom-partition")
        .data(
            {
                "weights": [11, 7, 9, 4, 6, 5, 3, 8],
                "k": 3,
                "max_group_load": 20,
            }
        )
        .constructor(make_initial)
        .objective(evaluate, hard_violation_weight=1_000.0)
        .moves(generate_moves, apply_move)
        .build()
    )

    result = AutoSolver(time_limit=2.0, seed=21).solve(problem)
    print(result.summary())
    print("best:", result.best_solution.to_dict())


if __name__ == "__main__":
    main()
