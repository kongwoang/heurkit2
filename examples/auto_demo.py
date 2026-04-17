#!/usr/bin/env python3
"""
AutoSolver Demo — use the automatic solver across all 3 problem types.

Usage:
    python examples/auto_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.kernels.cvrp.problem import CVRPProblem
from heurkit.kernels.binpacking.problem import BinPackingProblem
from heurkit.portfolio.auto import AutoSolver


def main() -> None:
    print("=" * 60)
    print("  HeurKit — AutoSolver Demo")
    print("=" * 60)

    # Show available presets
    print("\nAvailable presets:")
    for pt, algos in AutoSolver.available_presets().items():
        print(f"  {pt}: {', '.join(algos)}")

    problems = [
        TSPProblem.generate_random(n_cities=25, seed=42),
        CVRPProblem.generate_random(n_customers=15, capacity=50.0, seed=42),
        BinPackingProblem.generate_random(n_items=30, capacity=100.0, seed=42),
    ]

    print()
    for problem in problems:
        # Simple one-liner API
        result = AutoSolver(time_limit=3.0, seed=42).solve(problem)

        feas = "✓" if result.is_feasible else "✗"
        print(f"  {problem.name():<20s}  obj={result.best_objective:>10.2f}  {feas}"
              f"  by {result.algorithm_name}  ({result.runtime_seconds:.2f}s)")

    # Example with custom picks
    print("\n  Custom picks (TabuSearch only on TSP):")
    tsp = TSPProblem.generate_random(n_cities=20, seed=99)
    result = AutoSolver(time_limit=2.0, seed=99, picks=["TabuSearch"]).solve(tsp)
    print(f"    {tsp.name()}: obj={result.best_objective:.2f} by {result.algorithm_name}")

    print("\n✓ AutoSolver demo complete.")


if __name__ == "__main__":
    main()
