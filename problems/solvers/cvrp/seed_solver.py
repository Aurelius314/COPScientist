"""Capacity-feasible greedy nearest-neighbor CVRP seed, as used by DRAGON."""

import math


def solve(instance):
    """Return routes containing the original customer IDs, without depot visits."""
    coordinates = instance["coordinates"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    depot = instance["depot"]
    unvisited = set(coordinates) - {depot}
    routes = []
    while unvisited:
        route = []
        remaining_capacity = capacity
        current = depot
        while True:
            feasible = (node for node in unvisited if demands[node] <= remaining_capacity)
            next_node = min(
                feasible,
                key=lambda node: (
                    math.floor(math.dist(coordinates[current], coordinates[node]) + 0.5),
                    node,
                ),
                default=None,
            )
            if next_node is None:
                break
            route.append(next_node)
            unvisited.remove(next_node)
            remaining_capacity -= demands[next_node]
            current = next_node
        if not route:
            raise ValueError("Customer demand exceeds vehicle capacity")
        routes.append(route)
    return routes
