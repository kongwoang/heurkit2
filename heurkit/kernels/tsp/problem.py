"""
TSP problem definition.

Holds city coordinates and/or a precomputed distance matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from heurkit.core.problem import Problem
from heurkit.kernels.tsp.evaluator import TSPEvaluator
from heurkit.kernels.tsp.constructors import NearestNeighborConstructor
from heurkit.kernels.tsp.neighbors import TSPNeighborhood


class TSPProblem(Problem):
    """Symmetric TSP instance.

    Parameters
    ----------
    distance_matrix : NDArray
        Square symmetric matrix of pairwise distances.
    coordinates : NDArray or None
        (n, 2) array of city coordinates (kept for plotting).
    instance_name : str
        Optional name for logging.

    Raises
    ------
    ValueError
        If the distance matrix is not square, or has fewer than 3 cities.
    """

    def __init__(
        self,
        distance_matrix: NDArray,
        coordinates: NDArray | None = None,
        instance_name: str = "TSP",
    ) -> None:
        distance_matrix = np.asarray(distance_matrix, dtype=float)
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(
                f"Distance matrix must be square, got shape {distance_matrix.shape}"
            )
        if distance_matrix.shape[0] < 3:
            raise ValueError(
                f"TSP requires at least 3 cities, got {distance_matrix.shape[0]}"
            )
        self.distance_matrix = distance_matrix
        self.coordinates = coordinates
        self.instance_name = instance_name
        self.n_cities: int = distance_matrix.shape[0]

    # ---- factory helpers ---------------------------------------------------

    @classmethod
    def from_coordinates(
        cls,
        coords: NDArray | list[tuple[float, float]],
        instance_name: str = "TSP",
    ) -> TSPProblem:
        """Create a TSP from 2-D coordinates using Euclidean distances."""
        coords = np.asarray(coords, dtype=float)
        n = len(coords)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        return cls(dist, coordinates=coords, instance_name=instance_name)

    @classmethod
    def from_distance_matrix(
        cls, matrix: NDArray, instance_name: str = "TSP"
    ) -> TSPProblem:
        return cls(np.asarray(matrix, dtype=float), instance_name=instance_name)

    @classmethod
    def generate_random(
        cls, n_cities: int = 20, seed: int | None = None
    ) -> TSPProblem:
        """Generate a random TSP in [0, 100]²."""
        rng = np.random.default_rng(seed)
        coords = rng.random((n_cities, 2)) * 100
        return cls.from_coordinates(coords, instance_name=f"TSP-rand-{n_cities}")

    # ---- Problem interface -------------------------------------------------

    def name(self) -> str:
        return self.instance_name

    def size(self) -> int:
        return self.n_cities

    # ---- kernel defaults ---------------------------------------------------

    def default_evaluator(self) -> TSPEvaluator:
        return TSPEvaluator(self)

    def default_constructor(self) -> NearestNeighborConstructor:
        return NearestNeighborConstructor(self)

    def default_neighborhood(self) -> TSPNeighborhood:
        return TSPNeighborhood(self)
