#!/usr/bin/env python3
"""Minimal custom-problem demo for HeurKit's low-code API."""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.custom import ProblemBuilder
from heurkit.portfolio.auto import AutoSolver


@dataclass
class State:
    value: int


def make_initial(problem_data, rng):
    return State(value=problem_data["start"])


def evaluate(problem_data, state):
    return float(abs(state.value - problem_data["target"]))


def generate_moves(problem_data, state, rng):
    yield -1
    yield 1


def apply_move(problem_data, state, move):
    lo = problem_data["min"]
    hi = problem_data["max"]
    nxt = max(lo, min(hi, state.value + int(move)))
    return State(value=nxt)


def main() -> None:
    problem = (
        ProblemBuilder("minimal-custom")
        .data({"start": 12, "target": 0, "min": -20, "max": 20})
        .constructor(make_initial)
        .objective(evaluate)
        .moves(generate_moves, apply_move)
        .build()
    )

    result = AutoSolver(time_limit=1.0, seed=42).solve(problem)
    print(result.summary())
    print("best_state:", result.best_solution.to_dict())


if __name__ == "__main__":
    main()
