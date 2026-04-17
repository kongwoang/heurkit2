"""Tests for the Bin Packing kernel."""

import pytest
import numpy as np

from heurkit.kernels.binpacking.problem import BinPackingProblem
from heurkit.kernels.binpacking.solution import BinPackingSolution
from heurkit.kernels.binpacking.evaluator import BinPackingEvaluator
from heurkit.kernels.binpacking.constructors import (
    FirstFitConstructor,
    FirstFitDecreasingConstructor,
)
from heurkit.kernels.binpacking.neighbors import BinPackingNeighborhood
from heurkit.kernels.binpacking.moves import MoveItemMove, SwapItemsMove


@pytest.fixture
def bp_problem():
    return BinPackingProblem.generate_random(n_items=20, capacity=100.0, seed=42)


class TestBinPackingProblem:
    def test_from_sizes(self):
        p = BinPackingProblem.from_sizes([10, 20, 30, 40, 50], capacity=60)
        assert p.n_items == 5
        assert p.bin_capacity == 60.0

    def test_generate_random(self):
        p = BinPackingProblem.generate_random(n_items=25, seed=99)
        assert p.n_items == 25
        assert all(s <= p.bin_capacity * 0.6 for s in p.item_sizes)


class TestBinPackingConstructors:
    def test_first_fit_assigns_all(self, bp_problem):
        c = FirstFitConstructor(bp_problem)
        sol = c.construct(bp_problem)
        assert isinstance(sol, BinPackingSolution)
        all_items = {item for b in sol.bins for item in b}
        assert all_items == set(range(bp_problem.n_items))

    def test_first_fit_decreasing_assigns_all(self, bp_problem):
        c = FirstFitDecreasingConstructor(bp_problem)
        sol = c.construct(bp_problem)
        all_items = {item for b in sol.bins for item in b}
        assert all_items == set(range(bp_problem.n_items))

    def test_ffd_uses_fewer_or_equal_bins_than_ff(self, bp_problem):
        ff = FirstFitConstructor(bp_problem).construct(bp_problem)
        ffd = FirstFitDecreasingConstructor(bp_problem).construct(bp_problem)
        assert len(ffd.bins) <= len(ff.bins)


class TestBinPackingEvaluator:
    def test_evaluate_feasible(self, bp_problem):
        c = FirstFitDecreasingConstructor(bp_problem)
        sol = c.construct(bp_problem)
        ev = BinPackingEvaluator(bp_problem)
        result = ev.evaluate(sol)
        assert result.is_feasible is True
        assert result.objective == len(sol.bins)

    def test_overloaded_bin_is_infeasible(self, bp_problem):
        # Put all items in one bin — likely over capacity
        sol = BinPackingSolution([list(range(bp_problem.n_items))])
        ev = BinPackingEvaluator(bp_problem)
        result = ev.evaluate(sol)
        total = sum(bp_problem.item_sizes)
        if total > bp_problem.bin_capacity:
            assert result.is_feasible == False


class TestBinPackingMoves:
    def test_move_item(self, bp_problem):
        c = FirstFitDecreasingConstructor(bp_problem)
        sol = c.construct(bp_problem)
        original_items = {item for b in sol.bins for item in b}

        if len(sol.bins) >= 2 and len(sol.bins[0]) >= 1:
            move = MoveItemMove(0, 0, 1)
            move.apply(sol)
            new_items = {item for b in sol.bins for item in b}
            assert new_items == original_items

    def test_swap_items(self, bp_problem):
        c = FirstFitDecreasingConstructor(bp_problem)
        sol = c.construct(bp_problem)
        original_items = {item for b in sol.bins for item in b}

        if len(sol.bins) >= 2 and len(sol.bins[0]) >= 1 and len(sol.bins[1]) >= 1:
            move = SwapItemsMove(0, 0, 1, 0)
            move.apply(sol)
            new_items = {item for b in sol.bins for item in b}
            assert new_items == original_items


class TestBinPackingNeighborhood:
    def test_generates_moves(self, bp_problem):
        c = FirstFitDecreasingConstructor(bp_problem)
        sol = c.construct(bp_problem)
        nbr = BinPackingNeighborhood(bp_problem, seed=0)
        moves = list(nbr.generate(sol))
        assert len(moves) > 0
