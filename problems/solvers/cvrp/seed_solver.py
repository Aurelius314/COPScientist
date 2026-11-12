"""Feasible nearest-neighbor CVRP baseline using original TSPLIB node IDs."""

import math


def solve(instance):
    coords = instance["coordinates"]
    demands = instance["demands"]
    remaining = set(coords) - {instance["depot"]}
    routes = []
    while remaining:
        route, load, current = [], 0, instance["depot"]
        while True:
            feasible = [node for node in remaining if load + demands[node] <= instance["capacity"]]
            if not feasible:
                break
            nxt = min(feasible, key=lambda node: math.dist(coords[current], coords[node]))
            route.append(nxt)
            remaining.remove(nxt)
            load += demands[nxt]
            current = nxt
        if not route:
            raise ValueError("Customer demand exceeds capacity")
        routes.append(route)
    return routes
