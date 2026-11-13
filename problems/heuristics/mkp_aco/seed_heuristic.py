"""Seed heuristic for ACO-MKP."""

import numpy as np


def heuristics(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    load = weight.sum(axis=0)
    bind_density = prize / np.max(weight * load, axis=1)
    sum_density = prize / np.sum(weight, axis=1)
    return bind_density * sum_density
