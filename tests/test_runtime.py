"""Tests for the generic runtime and core abstractions."""

import pytest

from heurkit.core.stopping import StoppingCriteria
from heurkit.core.result import SearchResult
from heurkit.core.evaluator import Evaluation, Evaluator


class TestStoppingCriteria:
    def test_max_iterations(self):
        sc = StoppingCriteria(max_iterations=3)
        sc.start()
        assert not sc.should_stop()
        sc.step(False)
        sc.step(False)
        sc.step(False)
        assert sc.should_stop()

    def test_no_improvement(self):
        sc = StoppingCriteria(no_improvement_iterations=2)
        sc.start()
        sc.step(True)
        assert not sc.should_stop()
        sc.step(False)
        sc.step(False)
        assert sc.should_stop()

    def test_improvement_resets_counter(self):
        sc = StoppingCriteria(no_improvement_iterations=3)
        sc.start()
        sc.step(False)
        sc.step(False)
        sc.step(True)  # reset
        assert not sc.should_stop()
        sc.step(False)
        assert not sc.should_stop()

    def test_iteration_tracking(self):
        sc = StoppingCriteria(max_iterations=100)
        sc.start()
        for _ in range(10):
            sc.step(False)
        assert sc.iteration == 10


class TestSearchResult:
    def test_summary(self):
        from heurkit.kernels.tsp.solution import TSPSolution
        sol = TSPSolution([0, 1, 2])
        r = SearchResult(
            algorithm_name="TestAlgo",
            problem_name="TestProblem",
            best_solution=sol,
            best_objective=42.5,
            is_feasible=True,
            iterations=100,
            runtime_seconds=1.23,
            history=[50, 45, 42.5],
        )
        s = r.summary()
        assert "TestAlgo" in s
        assert "42.5" in s
        assert r.iterations == 100


class TestEvaluation:
    def test_is_better_minimisation(self):
        e1 = Evaluation(objective=10.0, is_feasible=True)
        e2 = Evaluation(objective=20.0, is_feasible=True)
        ev = _DummyEvaluator()
        assert ev.is_better(e1, e2)
        assert not ev.is_better(e2, e1)

    def test_feasible_beats_infeasible(self):
        feasible = Evaluation(objective=100.0, is_feasible=True)
        infeasible = Evaluation(objective=10.0, is_feasible=False)
        ev = _DummyEvaluator()
        assert ev.is_better(feasible, infeasible)
        assert not ev.is_better(infeasible, feasible)


class _DummyEvaluator(Evaluator):
    def evaluate(self, solution):
        return Evaluation(objective=0, is_feasible=True)
