"""Tests for input validation across all problem kernels."""

import pytest
import numpy as np

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.kernels.cvrp.problem import CVRPProblem
from heurkit.kernels.binpacking.problem import BinPackingProblem


class TestTSPValidation:
    def test_rejects_non_square_matrix(self):
        with pytest.raises(ValueError, match="square"):
            TSPProblem(np.zeros((3, 4)))

    def test_rejects_too_few_cities(self):
        with pytest.raises(ValueError, match="at least 3"):
            TSPProblem(np.zeros((2, 2)))

    def test_accepts_valid_matrix(self):
        p = TSPProblem(np.zeros((5, 5)))
        assert p.n_cities == 5


class TestCVRPValidation:
    def test_rejects_zero_capacity(self):
        with pytest.raises(ValueError, match="positive"):
            CVRPProblem(np.zeros((3, 3)), np.array([0, 5, 5]), capacity=0)

    def test_rejects_mismatched_demands(self):
        with pytest.raises(ValueError, match="Demands length"):
            CVRPProblem(np.zeros((3, 3)), np.array([0, 5]), capacity=50)

    def test_rejects_demand_exceeding_capacity(self):
        with pytest.raises(ValueError, match="exceed"):
            CVRPProblem(np.zeros((3, 3)), np.array([0, 100, 5]), capacity=50)

    def test_accepts_valid_instance(self):
        p = CVRPProblem(np.zeros((3, 3)), np.array([0, 10, 20]), capacity=50)
        assert p.n_customers == 2


class TestBinPackingValidation:
    def test_rejects_zero_capacity(self):
        with pytest.raises(ValueError, match="positive"):
            BinPackingProblem(np.array([10, 20]), bin_capacity=0)

    def test_rejects_empty_items(self):
        with pytest.raises(ValueError, match="[Aa]t least one"):
            BinPackingProblem(np.array([]), bin_capacity=100)

    def test_rejects_negative_item(self):
        with pytest.raises(ValueError, match="positive"):
            BinPackingProblem(np.array([-5, 20]), bin_capacity=100)

    def test_rejects_oversized_item(self):
        with pytest.raises(ValueError, match="exceeds"):
            BinPackingProblem(np.array([10, 200]), bin_capacity=100)

    def test_accepts_valid_instance(self):
        p = BinPackingProblem(np.array([10, 20, 30]), bin_capacity=100)
        assert p.n_items == 3
