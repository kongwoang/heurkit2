"""Tests for the TSP kernel."""

import pytest
import numpy as np

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.kernels.tsp.solution import TSPSolution
from heurkit.kernels.tsp.evaluator import TSPEvaluator
from heurkit.kernels.tsp.constructors import RandomConstructor, NearestNeighborConstructor
from heurkit.kernels.tsp.neighbors import TSPNeighborhood
from heurkit.kernels.tsp.moves import SwapMove, TwoOptMove, InsertMove


@pytest.fixture
def tsp_problem():
    return TSPProblem.generate_random(n_cities=10, seed=42)


class TestTSPProblem:
    def test_from_coordinates(self):
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        p = TSPProblem.from_coordinates(coords)
        assert p.n_cities == 4
        assert p.distance_matrix.shape == (4, 4)

    def test_generate_random(self):
        p = TSPProblem.generate_random(n_cities=15, seed=99)
        assert p.n_cities == 15
        assert p.coordinates is not None
        assert p.coordinates.shape == (15, 2)


class TestTSPConstructors:
    def test_random_constructor_valid(self, tsp_problem):
        c = RandomConstructor(tsp_problem, seed=0)
        sol = c.construct(tsp_problem)
        assert isinstance(sol, TSPSolution)
        assert sorted(sol.tour) == list(range(tsp_problem.n_cities))

    def test_nearest_neighbor_valid(self, tsp_problem):
        c = NearestNeighborConstructor(tsp_problem)
        sol = c.construct(tsp_problem)
        assert isinstance(sol, TSPSolution)
        assert sorted(sol.tour) == list(range(tsp_problem.n_cities))


class TestTSPEvaluator:
    def test_evaluate_returns_evaluation(self, tsp_problem):
        c = NearestNeighborConstructor(tsp_problem)
        sol = c.construct(tsp_problem)
        ev = TSPEvaluator(tsp_problem)
        result = ev.evaluate(sol)
        assert result.objective > 0
        assert result.is_feasible is True

    def test_known_tour_distance(self):
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        p = TSPProblem.from_coordinates(coords)
        sol = TSPSolution([0, 1, 2, 3])
        ev = TSPEvaluator(p)
        result = ev.evaluate(sol)
        assert abs(result.objective - 4.0) < 1e-6  # unit square perimeter


class TestTSPMoves:
    def test_swap_preserves_validity(self, tsp_problem):
        sol = TSPSolution(list(range(tsp_problem.n_cities)))
        move = SwapMove(0, 5)
        move.apply(sol)
        assert sorted(sol.tour) == list(range(tsp_problem.n_cities))

    def test_two_opt_preserves_validity(self, tsp_problem):
        sol = TSPSolution(list(range(tsp_problem.n_cities)))
        move = TwoOptMove(2, 7)
        move.apply(sol)
        assert sorted(sol.tour) == list(range(tsp_problem.n_cities))

    def test_insert_preserves_validity(self, tsp_problem):
        sol = TSPSolution(list(range(tsp_problem.n_cities)))
        move = InsertMove(3, 8)
        move.apply(sol)
        assert sorted(sol.tour) == list(range(tsp_problem.n_cities))


class TestTSPNeighborhood:
    def test_generates_moves(self, tsp_problem):
        sol = TSPSolution(list(range(tsp_problem.n_cities)))
        nbr = TSPNeighborhood(tsp_problem, seed=0)
        moves = list(nbr.generate(sol))
        assert len(moves) > 0
