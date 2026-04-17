#!/usr/bin/env python3
"""
Benchmark Demo — compare all algorithms across all 3 problem types.

Uses the benchmark runner for structured output with CSV/JSON export.

Usage:
    python examples/compare_algorithms.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.kernels.cvrp.problem import CVRPProblem
from heurkit.kernels.binpacking.problem import BinPackingProblem
from heurkit.algorithms.greedy import GreedyConstructor
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.algorithms.vns import VariableNeighborhoodSearch
from heurkit.portfolio.auto import AutoSolver
from heurkit.benchmark.runner import BenchmarkConfig, run_benchmark


SEED = 42
TIME = 1.5


def main() -> None:
    print("=" * 70)
    print("  HeurKit — Cross-Domain Algorithm Benchmark")
    print("=" * 70)

    problems = [
        TSPProblem.generate_random(n_cities=25, seed=SEED),
        CVRPProblem.generate_random(n_customers=15, capacity=50.0, seed=SEED),
        BinPackingProblem.generate_random(n_items=30, capacity=100.0, seed=SEED),
    ]

    algorithms = [
        GreedyConstructor(seed=SEED),
        HillClimbing(time_limit=TIME, seed=SEED),
        SimulatedAnnealing(time_limit=TIME, seed=SEED),
        TabuSearch(time_limit=TIME, seed=SEED),
        IteratedLocalSearch(time_limit=TIME, seed=SEED),
        VariableNeighborhoodSearch(time_limit=TIME, seed=SEED),
    ]

    os.makedirs("output", exist_ok=True)

    config = BenchmarkConfig(
        name="cross_domain",
        problems=problems,
        algorithms=algorithms,
        output_dir="output",
    )

    print("\nRunning benchmark...")
    bench = run_benchmark(config)

    print(f"\nCompleted in {bench.wall_time:.2f}s\n")
    print(bench.summary_table())

    # Also run AutoSolver for comparison
    print("\n--- AutoSolver ---")
    for problem in problems:
        result = AutoSolver(time_limit=TIME, seed=SEED).solve(problem)
        feas = "✓" if result.is_feasible else "✗"
        print(f"  {problem.name():<20s}  obj={result.best_objective:>10.2f}  {feas}  by {result.algorithm_name}")

    # Convergence plots
    try:
        from heurkit.utils.plotting import plot_convergence
        # Group results by problem
        for problem in problems:
            pname = problem.name()
            prob_results = [r for r in bench.results if r.problem_name == pname and len(r.history) > 2]
            if prob_results:
                plot_convergence(
                    prob_results,
                    title=f"{pname} — Convergence",
                    save_path=f"output/{pname.lower().replace('-','_')}_convergence.png",
                )
    except Exception as e:
        print(f"\n(Plotting skipped: {e})")

    print(f"\n✓ Results saved to output/cross_domain.csv and output/cross_domain.json")


if __name__ == "__main__":
    main()
