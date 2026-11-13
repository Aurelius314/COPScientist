"""Random-insertion TSP seed, following DRAGON's initialization heuristic."""

import math
import random


def solve(instance):
    """Return a zero-based tour using TSPLIB EUC_2D edge costs."""
    coordinates = instance["coordinates"]
    if len(coordinates) < 2:
        return list(range(len(coordinates)))

    def distance(a, b):
        x1, y1 = coordinates[a]
        x2, y2 = coordinates[b]
        return math.floor(math.hypot(x1 - x2, y1 - y2) + 0.5)

    # A local RNG makes repeated evaluations of the same instance comparable.
    unvisited = list(range(1, len(coordinates)))
    random.Random(0).shuffle(unvisited)
    tour = [0, unvisited.pop()]

    for city in unvisited:
        position = min(
            range(len(tour)),
            key=lambda index: distance(tour[index], city)
            + distance(city, tour[(index + 1) % len(tour)])
            - distance(tour[index], tour[(index + 1) % len(tour)]),
        )
        tour.insert(position + 1, city)
    return tour
