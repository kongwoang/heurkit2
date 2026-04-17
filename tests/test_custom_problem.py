"""Tests for the low-code custom problem layer."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.algorithms.iterated_local_search import IteratedLocalSearch
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing
from heurkit.algorithms.tabu import TabuSearch
from heurkit.algorithms.vns import VariableNeighborhoodSearch
from heurkit.custom import (
    CallbackExecutionError,
    CustomProblem,
    CustomProblemValidationError,
    ProblemBuilder,
    Score,
)
from heurkit.portfolio.auto import AutoSolver


@dataclass
class ScalarState:
    value: int


@dataclass
class AssignmentState:
    assignment: list[int]


def _build_scalar_problem() -> CustomProblem:
    def make_initial(problem_data, rng):
        return ScalarState(value=problem_data["start"])

    def evaluate(problem_data, state):
        return float(abs(state.value - problem_data["target"]))

    def generate_moves(problem_data, state, rng):
        yield -1
        yield 1

    def apply_move(problem_data, state, move):
        lo = problem_data["min"]
        hi = problem_data["max"]
        next_value = max(lo, min(hi, state.value + int(move)))
        return ScalarState(value=next_value)

    return (
        ProblemBuilder("scalar-min")
        .data({"start": 7, "target": 0, "min": -10, "max": 10})
        .constructor(make_initial)
        .objective(evaluate, sense="min")
        .moves(generate_moves, apply_move)
        .build()
    )


def _build_assignment_problem() -> CustomProblem:
    def make_initial(problem_data, rng):
        return AssignmentState(assignment=[0 for _ in problem_data["items"]])

    def evaluate(problem_data, state):
        loads = [0.0] * problem_data["num_bins"]
        for i, b in enumerate(state.assignment):
            loads[b] += problem_data["items"][i]

        over = sum(1.0 for load in loads if load > problem_data["capacity"])
        imbalance = max(loads) - min(loads)
        return Score(objective=max(loads), hard_violations=over, soft_penalty=imbalance)

    def is_feasible(problem_data, state):
        loads = [0.0] * problem_data["num_bins"]
        for i, b in enumerate(state.assignment):
            loads[b] += problem_data["items"][i]
        return all(load <= problem_data["capacity"] for load in loads)

    def generate_moves(problem_data, state, rng):
        for i in range(len(state.assignment)):
            for b in range(problem_data["num_bins"]):
                if b != state.assignment[i]:
                    yield (i, b)

    def apply_move(problem_data, state, move):
        i, b = move
        new_assignment = state.assignment.copy()
        new_assignment[i] = b
        return AssignmentState(new_assignment)

    return (
        ProblemBuilder("assignment-score")
        .data({
            "items": [4, 7, 2, 9],
            "num_bins": 3,
            "capacity": 11,
        })
        .constructor(make_initial)
        .objective(evaluate, sense="min", hard_violation_weight=500.0)
        .feasibility(is_feasible)
        .moves(generate_moves, apply_move)
        .build()
    )


class TestProblemBuilder:
    def test_builds_custom_problem(self):
        problem = _build_scalar_problem()
        assert isinstance(problem, CustomProblem)
        assert problem.name() == "scalar-min"
        assert problem.size() > 0

    def test_missing_required_callbacks_fails(self):
        with pytest.raises(CustomProblemValidationError, match="Missing required callbacks"):
            ProblemBuilder("bad").data({"x": 1}).build()

    def test_invalid_signature_fails(self):
        def bad_constructor(problem_data):
            return ScalarState(value=0)

        with pytest.raises(CustomProblemValidationError, match="constructor"):
            (
                ProblemBuilder("bad-signature")
                .data({"x": 1})
                .constructor(bad_constructor)
                .objective(lambda d, s: 1.0)
                .moves(lambda d, s, r: [0], lambda d, s, m: s)
                .build()
            )


class TestCustomExecution:
    def test_float_objective_runs(self):
        problem = _build_scalar_problem()
        result = HillClimbing(max_iterations=80, seed=3).solve(problem)
        assert result.problem_name == problem.name()
        assert result.best_objective <= 0.0
        assert result.is_feasible is True

    def test_score_objective_runs(self):
        problem = _build_assignment_problem()
        result = TabuSearch(max_iterations=120, seed=2).solve(problem)
        assert result.problem_name == problem.name()
        assert isinstance(result.best_objective, float)
        assert len(result.history) >= 1

    def test_moves_are_applied(self):
        problem = _build_scalar_problem()
        constructor = problem.default_constructor(seed=1)
        neighborhood = problem.default_neighborhood(seed=1)

        original = constructor.construct(problem)
        candidate = original.copy()

        moves = list(neighborhood.generate(candidate))
        assert len(moves) == 2

        before = candidate.state.value
        moves[0].apply(candidate)
        after = candidate.state.value
        assert after != before

    def test_objective_invalid_type_raises(self):
        problem = (
            ProblemBuilder("invalid-objective")
            .data({"x": 1})
            .constructor(lambda d, r: ScalarState(0))
            .objective(lambda d, s: "not-a-number")
            .moves(lambda d, s, r: [1], lambda d, s, m: ScalarState(s.value + m))
            .build()
        )

        with pytest.raises(CallbackExecutionError, match="objective"):
            HillClimbing(max_iterations=5, seed=0).solve(problem)

    def test_move_generator_invalid_raises(self):
        problem = (
            ProblemBuilder("invalid-moves")
            .data({"x": 1})
            .constructor(lambda d, r: ScalarState(0))
            .objective(lambda d, s: float(abs(s.value)))
            .moves(lambda d, s, r: 123, lambda d, s, m: ScalarState(s.value + 1))
            .build()
        )

        with pytest.raises(CallbackExecutionError, match="generate_moves"):
            HillClimbing(max_iterations=5, seed=0).solve(problem)

    def test_apply_move_invalid_state_raises(self):
        problem = (
            ProblemBuilder("invalid-apply")
            .data({"x": 1})
            .constructor(lambda d, r: ScalarState(0))
            .objective(lambda d, s: float(abs(s.value)))
            .moves(lambda d, s, r: [1], lambda d, s, m: None)
            .build()
        )

        with pytest.raises(CallbackExecutionError, match="apply_move"):
            HillClimbing(max_iterations=5, seed=0).solve(problem)


class TestAlgorithmsAndAutoSolver:
    @pytest.mark.parametrize(
        "algo",
        [
            HillClimbing(max_iterations=80, seed=1),
            TabuSearch(max_iterations=80, seed=1),
            SimulatedAnnealing(max_iterations=120, seed=1),
            IteratedLocalSearch(max_iterations=20, local_search_iters=25, seed=1),
            VariableNeighborhoodSearch(max_iterations=20, local_search_iters=25, seed=1),
        ],
        ids=lambda a: type(a).__name__,
    )
    def test_algorithms_work_on_custom_problem(self, algo):
        problem = _build_scalar_problem()
        result = algo.solve(problem)
        assert result.problem_name == problem.name()
        assert result.best_solution is not None
        assert isinstance(result.best_objective, float)

    def test_auto_solver_supports_custom_problem(self):
        problem = _build_assignment_problem()
        result = AutoSolver(time_limit=0.8, seed=11).solve(problem)

        assert result.metadata["problem_type"] == "custom"
        assert result.metadata["auto_solver"] is True
        assert result.algorithm_name in {
            "TabuSearch",
            "SimulatedAnnealing",
            "HillClimbing",
            "VNS",
            "IteratedLocalSearch",
        }

    def test_same_seed_is_deterministic_enough(self):
        def make_initial(problem_data, rng):
            bits = [int(rng.integers(0, 2)) for _ in range(problem_data["n"])]
            return AssignmentState(assignment=bits)

        def evaluate(problem_data, state):
            return float(sum(state.assignment))

        def generate_moves(problem_data, state, rng):
            # Seeded random neighborhood size/order.
            for _ in range(problem_data["n"]):
                yield int(rng.integers(0, problem_data["n"]))

        def apply_move(problem_data, state, move):
            out = state.assignment.copy()
            out[move] = 1 - out[move]
            return AssignmentState(out)

        problem = (
            ProblemBuilder("deterministic")
            .data({"n": 8})
            .constructor(make_initial)
            .objective(evaluate)
            .moves(generate_moves, apply_move)
            .with_seed(123)
            .build()
        )

        r1 = HillClimbing(max_iterations=60, seed=999).solve(problem)
        r2 = HillClimbing(max_iterations=60, seed=999).solve(problem)
        assert r1.best_objective == r2.best_objective

    def test_readme_style_flow_smoke(self):
        @dataclass
        class MyState:
            assignment: list[int]

        def make_initial(problem_data, rng):
            return MyState(assignment=[0 for _ in problem_data["items"]])

        def evaluate(problem_data, state):
            return float(sum(problem_data["items"][i] * b for i, b in enumerate(state.assignment)))

        def generate_moves(problem_data, state, rng):
            for i in range(len(state.assignment)):
                for b in range(problem_data["num_bins"]):
                    yield ("assign", i, b)

        def apply_move(problem_data, state, move):
            _, i, b = move
            new_state = MyState(assignment=state.assignment.copy())
            new_state.assignment[i] = b
            return new_state

        problem = (
            ProblemBuilder("my_custom_problem")
            .data({
                "items": [4, 7, 2, 9],
                "num_bins": 3,
                "capacity": 10,
            })
            .constructor(make_initial)
            .objective(evaluate, sense="min")
            .moves(generate_moves, apply_move)
            .build()
        )

        result = AutoSolver(time_limit=0.5, seed=42).solve(problem)
        assert isinstance(result.best_objective, float)
        assert result.best_solution is not None
