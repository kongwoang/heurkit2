#!/usr/bin/env python3
"""
TSP Demo — solve a random TSP instance with multiple algorithms.

Usage:
    python examples/tsp_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.algorithms.greedy import GreedyConstructor
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.utils.metrics import results_table


def main() -> None:
    print("=" * 60)
    print("  HeurKit — TSP Demo")
    print("=" * 60)

    # Generate a random TSP instance
    problem = TSPProblem.generate_random(n_cities=30, seed=42)
    print(f"\nProblem: {problem.name()} ({problem.size()} cities)\n")

    results = []

    # 1. Greedy constructor
    greedy = GreedyConstructor()
    result = greedy.solve(problem)
    results.append(result)
    print(f"  Greedy:       {result.best_objective:.2f}")

    # 2. Hill Climbing
    hc = HillClimbing(max_seconds=2.0, seed=42)
    result = hc.solve(problem)
    results.append(result)
    print(f"  Hill Climb:   {result.best_objective:.2f}")

    # 3. Simulated Annealing
    sa = SimulatedAnnealing(max_seconds=2.0, seed=42)
    result = sa.solve(problem)
    results.append(result)
    print(f"  SA:           {result.best_objective:.2f}")

    # 4. Tabu Search
    ts = TabuSearch(max_seconds=2.0, seed=42)
    result = ts.solve(problem)
    results.append(result)
    print(f"  Tabu:         {result.best_objective:.2f}")

    # 5. Iterated Local Search
    ils = IteratedLocalSearch(max_seconds=2.0, seed=42)
    result = ils.solve(problem)
    results.append(result)
    print(f"  ILS:          {result.best_objective:.2f}")

    print(f"\n{results_table(results)}")

    # Try to plot convergence (matplotlib optional)
    try:
        from heurkit.utils.plotting import plot_convergence
        plot_convergence(results, title="TSP Convergence", save_path="tsp_convergence.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
