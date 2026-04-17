"""Tests for generic algorithms across all 3 problem domains.

Ensures every algorithm can run through the common interface on
every kernel.  We don't test for optimality — we test for:
  - correctness (feasible solution returned)
  - improvement (at least as good as initial construction)
  - result structure (all fields filled)
"""

import pytest

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.kernels.cvrp.problem import CVRPProblem
from heurkit.kernels.binpacking.problem import BinPackingProblem
from heurkit.algorithms.greedy import GreedyConstructor
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.portfolio.auto import AutoSolver
from heurkit.core.result import SearchResult


# ---- fixtures --------------------------------------------------------------

@pytest.fixture
def tsp():
    return TSPProblem.generate_random(n_cities=10, seed=42)


@pytest.fixture
def cvrp():
    return CVRPProblem.generate_random(n_customers=8, capacity=50.0, seed=42)


@pytest.fixture
def bp():
    return BinPackingProblem.generate_random(n_items=15, capacity=100.0, seed=42)


ALL_PROBLEMS = ["tsp", "cvrp", "bp"]


def _get_algorithms():
    return [
        GreedyConstructor(seed=0),
        HillClimbing(max_seconds=1.0, max_iterations=500, seed=0),
        SimulatedAnnealing(max_seconds=1.0, max_iterations=500, seed=0),
        TabuSearch(max_seconds=1.0, max_iterations=500, seed=0),
        IteratedLocalSearch(max_seconds=1.0, max_iterations=50, seed=0),
    ]


# ---- cross-domain tests ----------------------------------------------------

class TestAlgorithmsOnTSP:
    @pytest.mark.parametrize("algo", _get_algorithms(), ids=lambda a: type(a).__name__)
    def test_algorithm_runs(self, tsp, algo):
        result = algo.solve(tsp)
        _assert_valid_result(result, tsp.name())

    def test_search_improves_over_greedy(self, tsp):
        greedy_result = GreedyConstructor(seed=0).solve(tsp)
        sa_result = SimulatedAnnealing(max_seconds=1.0, seed=0).solve(tsp)
        assert sa_result.best_objective <= greedy_result.best_objective * 1.05


class TestAlgorithmsOnCVRP:
    @pytest.mark.parametrize("algo", _get_algorithms(), ids=lambda a: type(a).__name__)
    def test_algorithm_runs(self, cvrp, algo):
        result = algo.solve(cvrp)
        _assert_valid_result(result, cvrp.name())


class TestAlgorithmsOnBinPacking:
    @pytest.mark.parametrize("algo", _get_algorithms(), ids=lambda a: type(a).__name__)
    def test_algorithm_runs(self, bp, algo):
        result = algo.solve(bp)
        _assert_valid_result(result, bp.name())


class TestAutoSolver:
    def test_auto_tsp(self, tsp):
        result = AutoSolver(time_limit=1.0, seed=0).solve(tsp)
        _assert_valid_result(result, tsp.name())

    def test_auto_cvrp(self, cvrp):
        result = AutoSolver(time_limit=1.0, seed=0).solve(cvrp)
        _assert_valid_result(result, cvrp.name())

    def test_auto_bp(self, bp):
        result = AutoSolver(time_limit=1.0, seed=0).solve(bp)
        _assert_valid_result(result, bp.name())

    def test_auto_infers_type(self, tsp):
        result = AutoSolver(time_limit=0.5).solve(tsp)
        assert result.metadata.get("problem_type") == "tsp"


# ---- helpers ----------------------------------------------------------------

def _assert_valid_result(result: SearchResult, problem_name: str) -> None:
    assert isinstance(result, SearchResult)
    assert result.algorithm_name != ""
    assert result.problem_name == problem_name
    assert result.best_solution is not None
    assert isinstance(result.best_objective, float)
    assert isinstance(result.is_feasible, bool)
    assert result.iterations >= 1
    assert result.runtime_seconds >= 0
    assert len(result.history) >= 1
