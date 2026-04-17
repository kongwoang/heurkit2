#!/usr/bin/env python3
"""
Compare Algorithms — benchmark all algorithms across all 3 problem types.

Usage:
    python examples/compare_algorithms.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.kernels.cvrp.problem import CVRPProblem
from heurkit.kernels.binpacking.problem import BinPackingProblem
from heurkit.algorithms.greedy import GreedyConstructor
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.portfolio.auto import AutoSolver
from heurkit.utils.metrics import results_table
from heurkit.core.result import SearchResult


def run_all_algorithms(problem, time_limit: float = 1.5, seed: int = 42) -> list[SearchResult]:
    """Run every algorithm on a problem and return results."""
    algorithms = [
        GreedyConstructor(seed=seed),
        HillClimbing(max_seconds=time_limit, seed=seed),
        SimulatedAnnealing(max_seconds=time_limit, seed=seed),
        TabuSearch(max_seconds=time_limit, seed=seed),
        IteratedLocalSearch(max_seconds=time_limit, seed=seed),
    ]
    results = []
    for algo in algorithms:
        result = algo.solve(problem)
        results.append(result)
    return results


def main() -> None:
    print("=" * 70)
    print("  HeurKit — Cross-Domain Algorithm Comparison")
    print("=" * 70)

    SEED = 42
    TIME = 1.5

    # --- TSP ---
    tsp = TSPProblem.generate_random(n_cities=25, seed=SEED)
    print(f"\n▶ {tsp.name()} ({tsp.size()} cities)")
    tsp_results = run_all_algorithms(tsp, time_limit=TIME, seed=SEED)

    # AutoSolver
    auto_result = AutoSolver(time_limit=TIME, seed=SEED).solve(tsp)
    auto_result.algorithm_name = "AutoSolver"
    tsp_results.append(auto_result)
    print(results_table(tsp_results))

    # --- CVRP ---
    cvrp = CVRPProblem.generate_random(n_customers=15, capacity=50.0, seed=SEED)
    print(f"\n▶ {cvrp.name()} ({cvrp.size()} customers)")
    cvrp_results = run_all_algorithms(cvrp, time_limit=TIME, seed=SEED)

    auto_result = AutoSolver(time_limit=TIME, seed=SEED).solve(cvrp)
    auto_result.algorithm_name = "AutoSolver"
    cvrp_results.append(auto_result)
    print(results_table(cvrp_results))

    # --- Bin Packing ---
    bp = BinPackingProblem.generate_random(n_items=30, capacity=100.0, seed=SEED)
    print(f"\n▶ {bp.name()} ({bp.size()} items)")
    bp_results = run_all_algorithms(bp, time_limit=TIME, seed=SEED)

    auto_result = AutoSolver(time_limit=TIME, seed=SEED).solve(bp)
    auto_result.algorithm_name = "AutoSolver"
    bp_results.append(auto_result)
    print(results_table(bp_results))

    # Convergence plots (optional)
    try:
        from heurkit.utils.plotting import plot_convergence
        plot_convergence(tsp_results[1:5], title="TSP Convergence", save_path="tsp_convergence.png")
        plot_convergence(cvrp_results[1:5], title="CVRP Convergence", save_path="cvrp_convergence.png")
        plot_convergence(bp_results[1:5], title="Bin Packing Convergence", save_path="bp_convergence.png")
        print("\nConvergence plots saved to *_convergence.png")
    except Exception as e:
        print(f"\n(Plotting skipped: {e})")

    print("\n✓ Benchmark complete.")


if __name__ == "__main__":
    main()
