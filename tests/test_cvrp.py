"""Tests for the CVRP kernel."""

import pytest
import numpy as np

from heurkit.kernels.cvrp.problem import CVRPProblem
from heurkit.kernels.cvrp.solution import CVRPSolution
from heurkit.kernels.cvrp.evaluator import CVRPEvaluator
from heurkit.kernels.cvrp.constructors import (
    GreedySequentialConstructor,
    RandomFeasibleConstructor,
)
from heurkit.kernels.cvrp.neighbors import CVRPNeighborhood
from heurkit.kernels.cvrp.moves import RelocateMove, SwapCustomersMove


@pytest.fixture
def cvrp_problem():
    return CVRPProblem.generate_random(n_customers=10, capacity=50.0, seed=42)


class TestCVRPProblem:
    def test_generate_random(self):
        p = CVRPProblem.generate_random(n_customers=8, seed=99)
        assert p.n_customers == 8
        assert p.demands[0] == 0.0  # depot demand
        assert p.capacity == 50.0

    def test_from_coordinates(self):
        p = CVRPProblem.from_coordinates(
            depot=(50, 50),
            customers=[(10, 10), (90, 90), (30, 70)],
            demands=[10, 20, 15],
            capacity=40,
        )
        assert p.n_customers == 3
        assert p.distance_matrix.shape == (4, 4)


class TestCVRPConstructors:
    def test_greedy_constructor_serves_all(self, cvrp_problem):
        c = GreedySequentialConstructor(cvrp_problem)
        sol = c.construct(cvrp_problem)
        assert isinstance(sol, CVRPSolution)
        assert sol.all_customers() == set(range(1, cvrp_problem.n_customers + 1))

    def test_random_feasible_constructor(self, cvrp_problem):
        c = RandomFeasibleConstructor(cvrp_problem, seed=0)
        sol = c.construct(cvrp_problem)
        assert sol.all_customers() == set(range(1, cvrp_problem.n_customers + 1))


class TestCVRPEvaluator:
    def test_evaluate_greedy_solution(self, cvrp_problem):
        c = GreedySequentialConstructor(cvrp_problem)
        sol = c.construct(cvrp_problem)
        ev = CVRPEvaluator(cvrp_problem)
        result = ev.evaluate(sol)
        assert result.objective > 0
        assert result.is_feasible is True

    def test_overloaded_route_is_infeasible(self, cvrp_problem):
        # Put all customers on one route — likely exceeds capacity
        all_custs = list(range(1, cvrp_problem.n_customers + 1))
        sol = CVRPSolution([all_custs])
        ev = CVRPEvaluator(cvrp_problem)
        result = ev.evaluate(sol)
        # With 10 customers (demands 5–19 each) and capacity 50, likely infeasible
        total_demand = sum(cvrp_problem.demands[c] for c in all_custs)
        if total_demand > cvrp_problem.capacity:
            assert result.is_feasible == False


class TestCVRPMoves:
    def test_swap_preserves_customers(self, cvrp_problem):
        c = GreedySequentialConstructor(cvrp_problem)
        sol = c.construct(cvrp_problem)
        original = sol.all_customers()

        if len(sol.routes) >= 2 and len(sol.routes[0]) >= 1 and len(sol.routes[1]) >= 1:
            move = SwapCustomersMove(0, 0, 1, 0)
            move.apply(sol)
            assert sol.all_customers() == original


class TestCVRPNeighborhood:
    def test_generates_moves(self, cvrp_problem):
        c = GreedySequentialConstructor(cvrp_problem)
        sol = c.construct(cvrp_problem)
        nbr = CVRPNeighborhood(cvrp_problem, seed=0)
        moves = list(nbr.generate(sol))
        assert len(moves) > 0
