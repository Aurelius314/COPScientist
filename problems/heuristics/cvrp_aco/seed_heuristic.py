"""Inverse-distance seed heuristic for ACO-CVRP."""

import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """Return the source seed heuristic, ``1 / distance_matrix``.

    """
    distances = np.asarray(distance_matrix, dtype=float)
    return 1.0 / np.maximum(distances, 1e-12)
