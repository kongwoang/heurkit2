#!/usr/bin/env python3
"""
Bin Packing Demo — solve a random 1D bin packing instance.

Usage:
    python examples/binpacking_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.kernels.binpacking.problem import BinPackingProblem
from heurkit.algorithms.greedy import GreedyConstructor
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.utils.metrics import results_table


def main() -> None:
    print("=" * 60)
    print("  HeurKit — Bin Packing Demo")
    print("=" * 60)

    # Generate a random bin packing instance
    problem = BinPackingProblem.generate_random(n_items=40, capacity=100.0, seed=42)
    print(f"\nProblem: {problem.name()} ({problem.size()} items, capacity={problem.bin_capacity})\n")

    results = []

    # 1. Greedy (First Fit Decreasing)
    greedy = GreedyConstructor()
    result = greedy.solve(problem)
    results.append(result)
    print(f"  FFD:          {result.best_objective:.0f} bins  (feasible={result.is_feasible})")

    # 2. Hill Climbing
    hc = HillClimbing(max_seconds=2.0, seed=42)
    result = hc.solve(problem)
    results.append(result)
    print(f"  Hill Climb:   {result.best_objective:.0f} bins  (feasible={result.is_feasible})")

    # 3. SA
    sa = SimulatedAnnealing(max_seconds=2.0, seed=42)
    result = sa.solve(problem)
    results.append(result)
    print(f"  SA:           {result.best_objective:.0f} bins  (feasible={result.is_feasible})")

    # 4. Tabu
    ts = TabuSearch(max_seconds=2.0, seed=42)
    result = ts.solve(problem)
    results.append(result)
    print(f"  Tabu:         {result.best_objective:.0f} bins  (feasible={result.is_feasible})")

    # 5. ILS
    ils = IteratedLocalSearch(max_seconds=2.0, seed=42)
    result = ils.solve(problem)
    results.append(result)
    print(f"  ILS:          {result.best_objective:.0f} bins  (feasible={result.is_feasible})")

    print(f"\n{results_table(results)}")


if __name__ == "__main__":
    main()
