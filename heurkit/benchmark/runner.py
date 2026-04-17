"""
Benchmark runner — compare algorithms across problem instances.

Provides deterministic, reproducible benchmarks with structured
output (tables, CSV, JSON).
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from heurkit.core.problem import Problem
from heurkit.core.result import SearchResult
from heurkit.core.runtime import SearchAlgorithm
from heurkit.utils.metrics import results_table


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes
    ----------
    name : str
        Human-readable name for this benchmark.
    problems : list[Problem]
        Problem instances to solve.
    algorithms : list[SearchAlgorithm]
        Algorithms to run on each problem.
    repetitions : int
        Number of repetitions per (problem, algorithm) pair.
    output_dir : str or None
        Directory for saving results.  Created if it doesn't exist.
    """

    name: str = "benchmark"
    problems: list[Problem] = field(default_factory=list)
    algorithms: list[SearchAlgorithm] = field(default_factory=list)
    repetitions: int = 1
    output_dir: str | None = None


@dataclass
class BenchmarkResult:
    """Collected results from a benchmark run."""

    config_name: str
    results: list[SearchResult] = field(default_factory=list)
    wall_time: float = 0.0

    def summary_table(self) -> str:
        """Return an ASCII table of all results."""
        return results_table(self.results)

    def to_records(self) -> list[dict[str, Any]]:
        """Return results as a list of flat dicts (for CSV/DataFrame)."""
        return [r.to_dict() for r in self.results]

    def save_csv(self, path: str) -> None:
        """Save results to a CSV file."""
        records = self.to_records()
        if not records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    def save_json(self, path: str) -> None:
        """Save results to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "benchmark": self.config_name,
            "wall_time_seconds": round(self.wall_time, 3),
            "n_results": len(self.results),
            "results": self.to_records(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Execute a benchmark run.

    Parameters
    ----------
    config : BenchmarkConfig
        The benchmark configuration.

    Returns
    -------
    BenchmarkResult
        Collected results with summary and export methods.
    """
    bench = BenchmarkResult(config_name=config.name)
    t0 = time.perf_counter()

    for problem in config.problems:
        for algo in config.algorithms:
            for rep in range(config.repetitions):
                result = algo.solve(problem)
                if config.repetitions > 1:
                    result.metadata["repetition"] = rep
                bench.results.append(result)

    bench.wall_time = time.perf_counter() - t0

    # Auto-save if output_dir is set
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        bench.save_csv(os.path.join(config.output_dir, f"{config.name}.csv"))
        bench.save_json(os.path.join(config.output_dir, f"{config.name}.json"))

    return bench
