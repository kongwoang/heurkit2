"""
Shared test fixtures for all test modules.
"""

import pytest

from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.kernels.cvrp.problem import CVRPProblem
from heurkit.kernels.binpacking.problem import BinPackingProblem


@pytest.fixture
def tsp():
    """Small TSP instance for fast tests."""
    return TSPProblem.generate_random(n_cities=10, seed=42)


@pytest.fixture
def cvrp():
    """Small CVRP instance for fast tests."""
    return CVRPProblem.generate_random(n_customers=8, capacity=50.0, seed=42)


@pytest.fixture
def bp():
    """Small bin packing instance for fast tests."""
    return BinPackingProblem.generate_random(n_items=15, capacity=100.0, seed=42)
