"""Nearest-neighbor TSP baseline. Nodes use zero-based coordinate order."""

import math


def solve(instance):
    coords = instance["coordinates"]
    remaining = set(range(1, len(coords)))
    tour = [0]
    while remaining:
        current = tour[-1]
        nxt = min(remaining, key=lambda node: math.dist(coords[current], coords[node]))
        tour.append(nxt)
        remaining.remove(nxt)
    return tour
