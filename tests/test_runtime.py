"""Tests for the generic runtime and core abstractions."""

import pytest

from heurkit.core.stopping import StoppingCriteria
from heurkit.core.result import SearchResult
from heurkit.core.evaluator import Evaluation, Evaluator
from heurkit.core.callbacks import HistoryCallback, LoggingCallback


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

    def test_elapsed_is_positive(self):
        sc = StoppingCriteria()
        sc.start()
        assert sc.elapsed >= 0


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

    def test_to_dict_keys(self):
        from heurkit.kernels.tsp.solution import TSPSolution
        sol = TSPSolution([0, 1, 2])
        r = SearchResult(
            algorithm_name="A", problem_name="P",
            best_solution=sol, best_objective=1.0,
            is_feasible=True, iterations=1, runtime_seconds=0.1,
        )
        d = r.to_dict()
        assert "algorithm_name" in d
        assert "best_objective" in d
        assert "is_feasible" in d
        assert isinstance(d["best_objective"], float)

    def test_to_json_roundtrip(self):
        import json
        from heurkit.kernels.tsp.solution import TSPSolution
        sol = TSPSolution([0, 1, 2])
        r = SearchResult(
            algorithm_name="A", problem_name="P",
            best_solution=sol, best_objective=1.0,
            is_feasible=True, iterations=1, runtime_seconds=0.1,
        )
        parsed = json.loads(r.to_json())
        assert parsed["algorithm_name"] == "A"


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

    def test_python_native_types(self):
        """Evaluation should coerce numpy types to Python native."""
        import numpy as np
        e = Evaluation(objective=np.float64(42.0), is_feasible=np.bool_(True))
        assert type(e.objective) is float
        assert type(e.is_feasible) is bool

    def test_is_identity_works(self):
        """After coercion, `is True` / `is False` should work."""
        e = Evaluation(objective=1.0, is_feasible=True)
        assert e.is_feasible is True
        e2 = Evaluation(objective=1.0, is_feasible=False)
        assert e2.is_feasible is False


class TestCallbackInstantiation:
    def test_history_callback(self):
        cb = HistoryCallback()
        assert cb.events == []

    def test_logging_callback_no_crash(self):
        cb = LoggingCallback(interval=100)
        # Just verify it can be created without errors
        assert cb.interval == 100


class _DummyEvaluator(Evaluator):
    def evaluate(self, solution):
        return Evaluation(objective=0, is_feasible=True)
