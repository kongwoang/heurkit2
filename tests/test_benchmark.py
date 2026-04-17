"""Smoke tests for the benchmark runner."""

import os
import tempfile

import pytest

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.algorithms.greedy import GreedyConstructor
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.benchmark.runner import BenchmarkConfig, BenchmarkResult, run_benchmark


class TestBenchmarkRunner:
    def test_basic_run(self):
        config = BenchmarkConfig(
            name="test",
            problems=[TSPProblem.generate_random(n_cities=8, seed=0)],
            algorithms=[GreedyConstructor(seed=0)],
        )
        result = run_benchmark(config)
        assert isinstance(result, BenchmarkResult)
        assert len(result.results) == 1
        assert result.wall_time > 0

    def test_multiple_algorithms(self):
        config = BenchmarkConfig(
            name="test_multi",
            problems=[TSPProblem.generate_random(n_cities=8, seed=0)],
            algorithms=[
                GreedyConstructor(seed=0),
                HillClimbing(max_iterations=50, seed=0),
            ],
        )
        result = run_benchmark(config)
        assert len(result.results) == 2

    def test_csv_export(self, tmp_path):
        config = BenchmarkConfig(
            name="csv_test",
            problems=[TSPProblem.generate_random(n_cities=8, seed=0)],
            algorithms=[GreedyConstructor(seed=0)],
            output_dir=str(tmp_path),
        )
        result = run_benchmark(config)
        csv_path = tmp_path / "csv_test.csv"
        assert csv_path.exists()

    def test_json_export(self, tmp_path):
        config = BenchmarkConfig(
            name="json_test",
            problems=[TSPProblem.generate_random(n_cities=8, seed=0)],
            algorithms=[GreedyConstructor(seed=0)],
            output_dir=str(tmp_path),
        )
        result = run_benchmark(config)
        json_path = tmp_path / "json_test.json"
        assert json_path.exists()
        import json
        data = json.loads(json_path.read_text())
        assert data["n_results"] == 1

    def test_summary_table(self):
        config = BenchmarkConfig(
            name="table_test",
            problems=[TSPProblem.generate_random(n_cities=8, seed=0)],
            algorithms=[GreedyConstructor(seed=0)],
        )
        result = run_benchmark(config)
        table = result.summary_table()
        assert "GreedyConstructor" in table
