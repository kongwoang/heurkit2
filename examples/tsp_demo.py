#!/usr/bin/env python3
"""
TSP Demo — solve a random TSP with multiple algorithms.

Usage:
    python examples/tsp_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.algorithms.greedy import GreedyConstructor
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.algorithms.vns import VariableNeighborhoodSearch
from heurkit.utils.metrics import results_table


def main() -> None:
    print("=" * 60)
    print("  HeurKit — TSP Demo")
    print("=" * 60)

    problem = TSPProblem.generate_random(n_cities=30, seed=42)
    print(f"\nProblem: {problem.name()} ({problem.size()} cities)\n")

    algorithms = [
        ("Greedy (NN)",  GreedyConstructor()),
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
        print(f"  {label:<14s}  obj={result.best_objective:>8.2f}  {feas}  ({result.runtime_seconds:.2f}s)")

    print(f"\n{results_table(results)}")

    # Save convergence plot
    try:
        from heurkit.utils.plotting import plot_convergence
        os.makedirs("output", exist_ok=True)
        plot_convergence(results[1:], title="TSP — Algorithm Convergence", save_path="output/tsp_convergence.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
