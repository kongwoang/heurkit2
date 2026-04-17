#!/usr/bin/env python3
"""Custom assignment/resource-allocation demo using callback API."""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.custom import ProblemBuilder, Score
from heurkit.portfolio.auto import AutoSolver


@dataclass
class AssignmentState:
    assignment: list[int]


def make_initial(problem_data, rng):
    return AssignmentState(assignment=[0 for _ in problem_data["items"]])


def evaluate(problem_data, state):
    loads = [0.0] * problem_data["num_bins"]
    for i, bin_idx in enumerate(state.assignment):
        loads[bin_idx] += problem_data["items"][i]

    max_load = max(loads)
    hard_violations = sum(1 for load in loads if load > problem_data["capacity"])
    balance_penalty = max_load - min(loads)
    return Score(
        objective=max_load,
        hard_violations=hard_violations,
        soft_penalty=balance_penalty,
    )


def is_feasible(problem_data, state):
    loads = [0.0] * problem_data["num_bins"]
    for i, bin_idx in enumerate(state.assignment):
        loads[bin_idx] += problem_data["items"][i]
    return all(load <= problem_data["capacity"] for load in loads)


def generate_moves(problem_data, state, rng):
    for i in range(len(state.assignment)):
        current_bin = state.assignment[i]
        for b in range(problem_data["num_bins"]):
            if b != current_bin:
                yield ("assign", i, b)


def apply_move(problem_data, state, move):
    _, i, b = move
    out = state.assignment.copy()
    out[i] = b
    return AssignmentState(assignment=out)


def pretty_print(problem_data, state):
    return f"assignment={state.assignment}"


def main() -> None:
    problem = (
        ProblemBuilder("custom-assignment")
        .data(
            {
                "items": [4, 7, 2, 9, 6, 5],
                "num_bins": 3,
                "capacity": 12,
            }
        )
        .constructor(make_initial)
        .objective(evaluate, sense="min", hard_violation_weight=1_000.0)
        .feasibility(is_feasible)
        .moves(generate_moves, apply_move)
        .pretty_print(pretty_print)
        .build()
    )

    result = AutoSolver(time_limit=2.0, seed=42).solve(problem)
    print(result.summary())
    print("best:", result.best_solution.to_dict())


if __name__ == "__main__":
    main()
