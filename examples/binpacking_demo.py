#!/usr/bin/env python3
"""
Bin Packing Demo — solve a random 1D bin packing instance.

Usage:
    python examples/binpacking_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.kernels.binpacking.problem import BinPackingProblem
from heurkit.algorithms.greedy import GreedyConstructor
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.algorithms.vns import VariableNeighborhoodSearch
from heurkit.utils.metrics import results_table


def main() -> None:
    print("=" * 60)
    print("  HeurKit — Bin Packing Demo")
    print("=" * 60)

    problem = BinPackingProblem.generate_random(n_items=40, capacity=100.0, seed=42)
    print(f"\nProblem: {problem.name()} ({problem.size()} items, capacity={problem.bin_capacity})\n")

    algorithms = [
        ("FFD",          GreedyConstructor()),
        ("Hill Climb",   HillClimbing(time_limit=2.0, seed=42)),
        ("Sim. Anneal.", SimulatedAnnealing(time_limit=2.0, seed=42)),
        ("Tabu Search",  TabuSearch(time_limit=2.0, seed=42)),
        ("ILS",          IteratedLocalSearch(time_limit=2.0, seed=42)),
        ("VNS",          VariableNeighborhoodSearch(time_limit=2.0, seed=42)),
    ]

    results = []
    for label, algo in algorithms:
        result = algo.solve(problem)
        results.append(result)
        feas = "✓" if result.is_feasible else "✗"
        print(f"  {label:<14s}  bins={result.best_objective:>4.0f}  {feas}  ({result.runtime_seconds:.2f}s)")

    print(f"\n{results_table(results)}")


if __name__ == "__main__":
    main()
